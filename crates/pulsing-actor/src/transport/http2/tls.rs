//! TLS support for HTTP/2 transport (passphrase-derived certificates).

use rcgen::{
    BasicConstraints, Certificate, CertificateParams, DnType, ExtendedKeyUsagePurpose, IsCa,
    KeyPair, KeyUsagePurpose, SerialNumber, PKCS_ED25519,
};
use ring::digest::{digest, SHA256};
use ring::hkdf::{self, HKDF_SHA256};
use ring::signature::{Ed25519KeyPair, KeyPair as RingKeyPair};
use rustls::crypto::ring::default_provider;
use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer, ServerName};
use rustls::server::WebPkiClientVerifier;
use rustls::{ClientConfig, RootCertStore, ServerConfig};
use std::sync::OnceLock;

static CRYPTO_PROVIDER_INSTALLED: OnceLock<()> = OnceLock::new();

fn ensure_crypto_provider() {
    CRYPTO_PROVIDER_INSTALLED.get_or_init(|| {
        let _ = default_provider().install_default();
    });
}
use std::sync::Arc;
use tokio_rustls::{TlsAcceptor, TlsConnector};

const HKDF_SALT: &[u8] = b"pulsing-ca-v1";

const CA_COMMON_NAME: &str = "Pulsing Cluster CA";

const NODE_CN_PREFIX: &str = "Pulsing Node";

const CERT_VALIDITY_SECS: i64 = 10 * 365 * 24 * 60 * 60;

/// TLS configuration for HTTP/2 transport.
#[derive(Clone)]
pub struct TlsConfig {
    pub acceptor: TlsAcceptor,
    pub connector: TlsConnector,
    passphrase_hash: String,
}

impl TlsConfig {
    /// Create TLS configuration from a passphrase.
    pub fn from_passphrase(passphrase: &str) -> anyhow::Result<Self> {
        ensure_crypto_provider();

        let (ca_cert, ca_key_pair) = derive_ca_from_passphrase(passphrase)?;

        let (node_cert, node_key_pair) = generate_node_cert(&ca_cert, &ca_key_pair)?;

        let ca_cert_der = CertificateDer::from(ca_cert.der().to_vec());
        let node_cert_der = CertificateDer::from(node_cert.der().to_vec());
        let node_key_der =
            PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(node_key_pair.serialize_der()));

        let mut root_store = RootCertStore::empty();
        root_store.add(ca_cert_der.clone())?;

        let client_verifier = WebPkiClientVerifier::builder(Arc::new(root_store.clone()))
            .build()
            .map_err(|e| anyhow::anyhow!("Failed to build client verifier: {}", e))?;

        let server_config = ServerConfig::builder()
            .with_client_cert_verifier(client_verifier)
            .with_single_cert(vec![node_cert_der.clone()], node_key_der.clone_key())
            .map_err(|e| anyhow::anyhow!("Failed to build server config: {}", e))?;

        let client_config = ClientConfig::builder()
            .with_root_certificates(root_store)
            .with_client_auth_cert(vec![node_cert_der], node_key_der)
            .map_err(|e| anyhow::anyhow!("Failed to build client config: {}", e))?;

        let hash = digest(&SHA256, passphrase.as_bytes());
        let passphrase_hash = hex_encode(&hash.as_ref()[..8]);

        Ok(Self {
            acceptor: TlsAcceptor::from(Arc::new(server_config)),
            connector: TlsConnector::from(Arc::new(client_config)),
            passphrase_hash,
        })
    }

    pub fn passphrase_hash(&self) -> &str {
        &self.passphrase_hash
    }

    /// Connect to a remote server with TLS.
    pub async fn connect(
        &self,
        stream: tokio::net::TcpStream,
        _server_name: &str,
    ) -> anyhow::Result<tokio_rustls::client::TlsStream<tokio::net::TcpStream>> {
        let server_name = ServerName::try_from("pulsing.internal".to_string())
            .map_err(|e| anyhow::anyhow!("Invalid server name: {}", e))?;

        self.connector
            .connect(server_name, stream)
            .await
            .map_err(|e| anyhow::anyhow!("TLS connect failed: {}", e))
    }

    pub async fn accept(
        &self,
        stream: tokio::net::TcpStream,
    ) -> anyhow::Result<tokio_rustls::server::TlsStream<tokio::net::TcpStream>> {
        self.acceptor
            .accept(stream)
            .await
            .map_err(|e| anyhow::anyhow!("TLS accept failed: {}", e))
    }
}

impl std::fmt::Debug for TlsConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TlsConfig")
            .field("passphrase_hash", &self.passphrase_hash)
            .finish()
    }
}

/// Derive CA certificate and key pair from passphrase.
fn derive_ca_from_passphrase(passphrase: &str) -> anyhow::Result<(Certificate, KeyPair)> {
    let seed = derive_seed(passphrase, b"ca-key")?;

    let key_pair = generate_deterministic_key_pair(&seed)?;

    let mut params = CertificateParams::new(vec![CA_COMMON_NAME.to_string()])
        .map_err(|e| anyhow::anyhow!("Failed to create cert params: {}", e))?;

    params
        .distinguished_name
        .push(DnType::CommonName, CA_COMMON_NAME);
    params
        .distinguished_name
        .push(DnType::OrganizationName, "Pulsing");
    params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
    params.key_usages = vec![
        KeyUsagePurpose::KeyCertSign,
        KeyUsagePurpose::CrlSign,
        KeyUsagePurpose::DigitalSignature,
    ];

    let not_before = time::OffsetDateTime::UNIX_EPOCH + time::Duration::days(365);
    let not_after = not_before + time::Duration::seconds(CERT_VALIDITY_SECS);
    params.not_before = not_before;
    params.not_after = not_after;

    // Derive serial number from seed for determinism
    let serial_seed = derive_seed(passphrase, b"ca-serial")?;
    params.serial_number = Some(SerialNumber::from_slice(&serial_seed[..20]));

    // Generate certificate
    let cert = params
        .self_signed(&key_pair)
        .map_err(|e| anyhow::anyhow!("Failed to generate CA cert: {}", e))?;

    Ok((cert, key_pair))
}

/// Internal server name used for TLS verification within the cluster
const INTERNAL_SERVER_NAME: &str = "pulsing.internal";

/// Generate a node certificate signed by the CA
fn generate_node_cert(
    ca_cert: &Certificate,
    ca_key: &KeyPair,
) -> anyhow::Result<(Certificate, KeyPair)> {
    // Generate a random key pair for the node (not deterministic)
    let node_key = KeyPair::generate_for(&PKCS_ED25519)
        .map_err(|e| anyhow::anyhow!("Failed to generate node key: {}", e))?;

    // Create node certificate
    let node_id = uuid::Uuid::new_v4().to_string();
    let cn = format!("{} {}", NODE_CN_PREFIX, &node_id[..8]);

    // Include internal server name as SAN for TLS verification
    let mut params = CertificateParams::new(vec![cn.clone(), INTERNAL_SERVER_NAME.to_string()])
        .map_err(|e| anyhow::anyhow!("Failed to create node cert params: {}", e))?;

    params.distinguished_name.push(DnType::CommonName, &cn);
    params
        .distinguished_name
        .push(DnType::OrganizationName, "Pulsing");
    params.is_ca = IsCa::NoCa;
    params.key_usages = vec![
        KeyUsagePurpose::DigitalSignature,
        KeyUsagePurpose::KeyEncipherment,
    ];
    params.extended_key_usages = vec![
        ExtendedKeyUsagePurpose::ServerAuth,
        ExtendedKeyUsagePurpose::ClientAuth,
    ];

    // Use current time for validity
    let not_before = time::OffsetDateTime::now_utc();
    let not_after = not_before + time::Duration::seconds(CERT_VALIDITY_SECS);
    params.not_before = not_before;
    params.not_after = not_after;

    // Generate certificate signed by CA
    let cert = params
        .signed_by(&node_key, ca_cert, ca_key)
        .map_err(|e| anyhow::anyhow!("Failed to sign node cert: {}", e))?;

    Ok((cert, node_key))
}

/// Helper struct for HKDF output length
struct HkdfLen(usize);

impl hkdf::KeyType for HkdfLen {
    fn len(&self) -> usize {
        self.0
    }
}

/// Derive a 32-byte seed using HKDF
fn derive_seed(passphrase: &str, info: &[u8]) -> anyhow::Result<[u8; 32]> {
    let salt = hkdf::Salt::new(HKDF_SHA256, HKDF_SALT);
    let prk = salt.extract(passphrase.as_bytes());

    let mut seed = [0u8; 32];
    prk.expand(&[info], HkdfLen(32))
        .map_err(|_| anyhow::anyhow!("HKDF expand failed"))?
        .fill(&mut seed)
        .map_err(|_| anyhow::anyhow!("HKDF fill failed"))?;

    Ok(seed)
}

/// Generate a deterministic Ed25519 key pair from a seed
///
/// Ed25519 natively supports deterministic key generation from a 32-byte seed.
fn generate_deterministic_key_pair(seed: &[u8; 32]) -> anyhow::Result<KeyPair> {
    // Create Ed25519 key pair from seed using ring
    let ed25519_key = Ed25519KeyPair::from_seed_unchecked(seed)
        .map_err(|e| anyhow::anyhow!("Failed to create Ed25519 key from seed: {}", e))?;

    // Get the PKCS#8 v2 DER encoding
    let pkcs8_der = create_ed25519_pkcs8_der(seed, ed25519_key.public_key().as_ref())?;

    // Convert to PrivateKeyDer and import into rcgen
    let private_key_der = PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(pkcs8_der));

    let key_pair = KeyPair::from_der_and_sign_algo(&private_key_der, &PKCS_ED25519)
        .map_err(|e| anyhow::anyhow!("Failed to create key pair from DER: {}", e))?;

    Ok(key_pair)
}

/// Create PKCS#8 v1 DER encoding for an Ed25519 private key
///
/// RFC 8410 defines the format for Ed25519 keys:
/// - privateKey contains CurvePrivateKey which is an OCTET STRING
/// - The 32-byte seed needs to be wrapped in an OCTET STRING
fn create_ed25519_pkcs8_der(seed: &[u8; 32], _public_key: &[u8]) -> anyhow::Result<Vec<u8>> {
    // OID for Ed25519: 1.3.101.112
    let ed25519_oid: &[u8] = &[0x06, 0x03, 0x2b, 0x65, 0x70];

    // Build algorithm identifier: SEQUENCE { OID ed25519 }
    let algo_seq = wrap_in_sequence(ed25519_oid);

    // CurvePrivateKey ::= OCTET STRING (the 32-byte seed)
    // This needs to be wrapped in OCTET STRING for the privateKey field
    let inner_private_key = wrap_in_octet_string(seed);
    let private_key_octet = wrap_in_octet_string(&inner_private_key);

    // Build PKCS#8 structure (version 0)
    let mut pkcs8_content = Vec::new();
    // Version INTEGER 0
    pkcs8_content.extend_from_slice(&[0x02, 0x01, 0x00]);
    // Algorithm identifier
    pkcs8_content.extend_from_slice(&algo_seq);
    // Private key (double-wrapped OCTET STRING)
    pkcs8_content.extend_from_slice(&private_key_octet);

    Ok(wrap_in_sequence(&pkcs8_content))
}

/// Wrap data in an ASN.1 SEQUENCE
fn wrap_in_sequence(data: &[u8]) -> Vec<u8> {
    let mut result = Vec::new();
    result.push(0x30); // SEQUENCE tag
    write_length(&mut result, data.len());
    result.extend_from_slice(data);
    result
}

/// Wrap data in an ASN.1 OCTET STRING
fn wrap_in_octet_string(data: &[u8]) -> Vec<u8> {
    let mut result = Vec::new();
    result.push(0x04); // OCTET STRING tag
    write_length(&mut result, data.len());
    result.extend_from_slice(data);
    result
}

/// Write ASN.1 DER length encoding
fn write_length(output: &mut Vec<u8>, len: usize) {
    if len < 128 {
        output.push(len as u8);
    } else if len < 256 {
        output.push(0x81);
        output.push(len as u8);
    } else {
        output.push(0x82);
        output.push((len >> 8) as u8);
        output.push(len as u8);
    }
}

/// Convert bytes to hex string
fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_derive_seed_deterministic() {
        let seed1 = derive_seed("test-password", b"test-info").unwrap();
        let seed2 = derive_seed("test-password", b"test-info").unwrap();
        assert_eq!(seed1, seed2);

        let seed3 = derive_seed("different-password", b"test-info").unwrap();
        assert_ne!(seed1, seed3);

        let seed4 = derive_seed("test-password", b"different-info").unwrap();
        assert_ne!(seed1, seed4);
    }

    #[test]
    fn test_ca_derivation_deterministic() {
        let (cert1, _) = derive_ca_from_passphrase("test-cluster-password").unwrap();
        let (cert2, _) = derive_ca_from_passphrase("test-cluster-password").unwrap();

        // Same passphrase should produce same CA certificate
        assert_eq!(cert1.der(), cert2.der());

        let (cert3, _) = derive_ca_from_passphrase("different-password").unwrap();
        // Different passphrase should produce different CA certificate
        assert_ne!(cert1.der(), cert3.der());
    }

    #[test]
    fn test_tls_config_creation() {
        let config = TlsConfig::from_passphrase("test-password");
        assert!(config.is_ok(), "TLS config creation failed: {:?}", config);
    }

    #[test]
    fn test_passphrase_hash() {
        let config = TlsConfig::from_passphrase("test-password").unwrap();
        let hash = config.passphrase_hash();
        // Hash should be consistent
        assert_eq!(hash.len(), 16); // 8 bytes = 16 hex chars
    }
}
