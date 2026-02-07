//! TLS support for HTTP/2 transport (passphrase-derived certificates).

use crate::error::{PulsingError, Result, RuntimeError};
use rcgen::{
    BasicConstraints, Certificate, CertificateParams, DnType, ExtendedKeyUsagePurpose, IsCa,
    KeyPair, KeyUsagePurpose, SerialNumber, PKCS_ED25519,
};
use rustls::crypto::aws_lc_rs::default_provider;
use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer, ServerName};
use rustls::server::WebPkiClientVerifier;
use rustls::{ClientConfig, RootCertStore, ServerConfig};
use sha2::{Digest, Sha256};
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
    pub fn from_passphrase(passphrase: &str) -> Result<Self> {
        ensure_crypto_provider();

        let (ca_cert, ca_key_pair) = derive_ca_from_passphrase(passphrase)
            .map_err(|e| PulsingError::from(RuntimeError::tls_error(e.to_string())))?;

        let (node_cert, node_key_pair) = generate_node_cert(&ca_cert, &ca_key_pair)
            .map_err(|e| PulsingError::from(RuntimeError::tls_error(e.to_string())))?;

        let ca_cert_der = CertificateDer::from(ca_cert.der().to_vec());
        let node_cert_der = CertificateDer::from(node_cert.der().to_vec());
        let node_key_der =
            PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(node_key_pair.serialize_der()));

        let mut root_store = RootCertStore::empty();
        root_store
            .add(ca_cert_der.clone())
            .map_err(|e| PulsingError::from(RuntimeError::tls_error(e.to_string())))?;

        let client_verifier = WebPkiClientVerifier::builder(Arc::new(root_store.clone()))
            .build()
            .map_err(|e| PulsingError::from(RuntimeError::tls_error(e.to_string())))?;

        let server_config = ServerConfig::builder()
            .with_client_cert_verifier(client_verifier)
            .with_single_cert(vec![node_cert_der.clone()], node_key_der.clone_key())
            .map_err(|e| PulsingError::from(RuntimeError::tls_error(e.to_string())))?;

        let client_config = ClientConfig::builder()
            .with_root_certificates(root_store)
            .with_client_auth_cert(vec![node_cert_der], node_key_der)
            .map_err(|e| PulsingError::from(RuntimeError::tls_error(e.to_string())))?;

        let hash = Sha256::digest(passphrase.as_bytes());
        let hash_slice = hash.as_slice();
        let hash_bytes = &hash_slice[..8];
        let passphrase_hash = hex_encode(hash_bytes);

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
    ) -> Result<tokio_rustls::client::TlsStream<tokio::net::TcpStream>> {
        let server_name = ServerName::try_from("pulsing.internal".to_string())
            .map_err(|e| PulsingError::from(RuntimeError::tls_error(e.to_string())))?;

        self.connector
            .connect(server_name, stream)
            .await
            .map_err(|e| PulsingError::from(RuntimeError::tls_error(e.to_string())))
    }

    pub async fn accept(
        &self,
        stream: tokio::net::TcpStream,
    ) -> Result<tokio_rustls::server::TlsStream<tokio::net::TcpStream>> {
        self.acceptor
            .accept(stream)
            .await
            .map_err(|e| PulsingError::from(RuntimeError::tls_error(e.to_string())))
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

/// Derive a 32-byte seed using SHA256
fn derive_seed(passphrase: &str, info: &[u8]) -> anyhow::Result<[u8; 32]> {
    let mut hasher = Sha256::new();
    hasher.update(HKDF_SALT);
    hasher.update(passphrase.as_bytes());
    hasher.update(info);
    let hash = hasher.finalize();

    let mut seed = [0u8; 32];
    seed.copy_from_slice(&hash);
    Ok(seed)
}

/// Generate a deterministic Ed25519 key pair from a seed
///
/// Manually constructs PKCS#8 DER format for Ed25519 keys per RFC 8410.
fn generate_deterministic_key_pair(seed: &[u8; 32]) -> anyhow::Result<KeyPair> {
    // Manually construct PKCS#8 DER for Ed25519 (RFC 8410)
    // PrivateKeyInfo ::= SEQUENCE {
    //   version INTEGER (0),
    //   algorithm AlgorithmIdentifier,
    //   privateKey OCTET STRING (contains OCTET STRING with seed)
    // }

    // Inner OCTET STRING containing the 32-byte seed
    let mut inner_private_key = Vec::new();
    inner_private_key.push(0x04); // OCTET STRING tag
    inner_private_key.push(32); // length
    inner_private_key.extend_from_slice(seed);

    // Outer OCTET STRING containing the inner OCTET STRING
    let mut outer_private_key = Vec::new();
    outer_private_key.push(0x04); // OCTET STRING tag
    outer_private_key.push(inner_private_key.len() as u8);
    outer_private_key.extend_from_slice(&inner_private_key);

    // Algorithm identifier for Ed25519 (OID 1.3.101.112)
    let algo_id: &[u8] = &[0x30, 0x05, 0x06, 0x03, 0x2b, 0x65, 0x70];

    // Build content: version + algorithm + privateKey
    let mut content = Vec::new();
    content.extend_from_slice(&[0x02, 0x01, 0x00]); // INTEGER 0 (version)
    content.extend_from_slice(algo_id); // AlgorithmIdentifier
    content.extend_from_slice(&outer_private_key); // privateKey

    // Wrap in SEQUENCE
    let mut pkcs8_der = Vec::new();
    pkcs8_der.push(0x30); // SEQUENCE tag
    pkcs8_der.push(content.len() as u8);
    pkcs8_der.extend_from_slice(&content);

    // Create rcgen KeyPair from the DER
    let private_key_der = rustls::pki_types::PrivatePkcs8KeyDer::from(pkcs8_der.as_slice());
    let key_pair = KeyPair::from_pkcs8_der_and_sign_algo(&private_key_der, &PKCS_ED25519)
        .map_err(|e| anyhow::anyhow!("Failed to create rcgen KeyPair: {}", e))?;

    Ok(key_pair)
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
