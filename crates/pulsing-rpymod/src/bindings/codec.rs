use std::cmp::min;

use futures::StreamExt;
use pulsing_actor::prelude::Message;
use pulsing_bindings_core::{
    reassemble_zerocopy_stream, zerocopy_mode, SEALED_PY_MSG_TYPE, SEALED_ZEROCOPY_MSG_TYPE,
    ZC_CHUNK_MSG_TYPE, ZC_DESCRIPTOR_MSG_TYPE, ZeroCopyDescriptorHeader,
};
use rustpython_vm::builtins::PyBytes;
use rustpython_vm::{AsObject, PyObjectRef, PyPayload, PyRef, PyResult, VirtualMachine};
use tokio::sync::mpsc;

use crate::interop::{pickle_object, unpickle_object};
use super::message::{PyMessage, PyZeroCopyDescriptor};

pub fn encode_python_payload(vm: &VirtualMachine, obj: &PyObjectRef) -> PyResult<Message> {
    match zerocopy_mode().as_str() {
        "off" => Ok(Message::single(SEALED_PY_MSG_TYPE, pickle_object(vm, obj)?)),
        "force" => {
            let zc = try_zerocopy_descriptor(vm, obj)?.ok_or_else(|| {
                vm.new_value_error("PULSING_ZEROCOPY=force but object does not provide __zerocopy__")
            })?;
            encode_zerocopy_message(vm, &zc)
        }
        _ => match try_zerocopy_descriptor(vm, obj)? {
            Some(zc) => encode_zerocopy_message(vm, &zc),
            None => Ok(Message::single(SEALED_PY_MSG_TYPE, pickle_object(vm, obj)?)),
        },
    }
}

fn try_zerocopy_descriptor(
    vm: &VirtualMachine,
    obj: &PyObjectRef,
) -> PyResult<Option<PyRef<PyZeroCopyDescriptor>>> {
    let zc_method = match obj.get_attr("__zerocopy__", vm) {
        Ok(m) => m,
        Err(_) => return Ok(None),
    };
    let builtins = vm.import("builtins", 0)?;
    let callable = vm.call_method(builtins.as_object(), "callable", (zc_method.clone(),))?;
    if !callable.try_into_value::<bool>(vm).unwrap_or(false) {
        return Ok(None);
    }
    let descriptor = zc_method.call((vm.ctx.none(),), vm)?;
    match descriptor.downcast_ref::<PyZeroCopyDescriptor>() {
        Some(d) => Ok(Some(d.to_owned())),
        None => Err(vm.new_value_error("__zerocopy__ must return ZeroCopyDescriptor")),
    }
}

fn encode_zerocopy_message(vm: &VirtualMachine, zc: &PyRef<PyZeroCopyDescriptor>) -> PyResult<Message> {
    let total = zc.total_buffer_bytes(vm);
    if total >= pulsing_bindings_core::message::zerocopy_stream_threshold() {
        encode_zerocopy_stream(vm, zc)
    } else {
        let bytes = zc.serialize_single(vm)?;
        Ok(Message::single(SEALED_ZEROCOPY_MSG_TYPE, bytes))
    }
}

fn encode_zerocopy_stream(vm: &VirtualMachine, zc: &PyRef<PyZeroCopyDescriptor>) -> PyResult<Message> {
    let chunk_len = pulsing_bindings_core::chunk_len();
    let header = zc.to_header(vm);
    let header_bytes = bincode::serialize(&header).map_err(|e| vm.new_value_error(e.to_string()))?;
    let buffer_data = zc.raw_buffer_data(vm)?;

    let (tx, rx) = mpsc::channel::<pulsing_actor::error::Result<Message>>(32);
    std::thread::spawn(move || {
        if tx
            .blocking_send(Ok(Message::single(ZC_DESCRIPTOR_MSG_TYPE, header_bytes)))
            .is_err()
        {
            return;
        }
        for buf in &buffer_data {
            let mut offset = 0;
            while offset < buf.len() {
                let end = min(offset + chunk_len, buf.len());
                let chunk = buf[offset..end].to_vec();
                if tx
                    .blocking_send(Ok(Message::single(ZC_CHUNK_MSG_TYPE, chunk)))
                    .is_err()
                {
                    return;
                }
                offset = end;
            }
        }
    });

    Ok(Message::from_channel(ZC_DESCRIPTOR_MSG_TYPE, rx))
}

pub async fn decode_message_to_pyobject(vm: &VirtualMachine, msg: Message) -> PyResult<PyObjectRef> {
    match msg {
        Message::Single {
            ref msg_type,
            ref data,
        } if msg_type == SEALED_PY_MSG_TYPE => unpickle_object(vm, data),
        Message::Single {
            ref msg_type,
            ref data,
        } if msg_type == SEALED_ZEROCOPY_MSG_TYPE => parse_zerocopy_single(vm, data),
        Message::Stream {
            ref default_msg_type,
            ..
        } if default_msg_type == ZC_DESCRIPTOR_MSG_TYPE => {
            let Message::Stream { mut stream, .. } = msg else {
                unreachable!()
            };
            let first = stream
                .next()
                .await
                .ok_or_else(|| vm.new_runtime_error("Empty zerocopy stream"))?
                .map_err(|e| vm.new_runtime_error(e.to_string()))?;
            let header_data = match first {
                Message::Single {
                    ref msg_type,
                    ref data,
                } if msg_type == ZC_DESCRIPTOR_MSG_TYPE => data.clone(),
                _ => {
                    return Err(vm.new_runtime_error(
                        "First frame of zerocopy stream must be descriptor",
                    ));
                }
            };
            let header: ZeroCopyDescriptorHeader =
                bincode::deserialize(&header_data).map_err(|e| vm.new_value_error(e.to_string()))?;
            let (header, raw_buffers) = reassemble_zerocopy_stream(header, &mut stream)
                .await
                .map_err(|e| vm.new_runtime_error(e.to_string()))?;
            let desc = PyZeroCopyDescriptor::from_wire(vm, header, raw_buffers);
            Ok(desc.into_ref(&vm.ctx).into())
        }
        other => {
            let py_msg = PyMessage::from_rust_message(other);
            Ok(py_msg.into_ref(&vm.ctx).into())
        }
    }
}

pub fn parse_zerocopy_single(vm: &VirtualMachine, data: &[u8]) -> PyResult<PyObjectRef> {
    if data.len() < 4 {
        return Err(vm.new_value_error("Zerocopy payload too short"));
    }
    let header_len = u32::from_le_bytes(data[..4].try_into().unwrap()) as usize;
    if data.len() < 4 + header_len {
        return Err(vm.new_value_error("Zerocopy payload truncated"));
    }
    let header: ZeroCopyDescriptorHeader =
        bincode::deserialize(&data[4..4 + header_len]).map_err(|e| vm.new_value_error(e.to_string()))?;
    let mut offset = 4 + header_len;
    let raw_buffers: Vec<Vec<u8>> = header
        .buffer_lengths
        .iter()
        .map(|&len| {
            let buf = data[offset..offset + len].to_vec();
            offset += len;
            buf
        })
        .collect();
    let desc = PyZeroCopyDescriptor::from_wire(vm, header, raw_buffers);
    Ok(desc.into_ref(&vm.ctx).into())
}

pub fn py_message_to_rust(vm: &VirtualMachine, obj: &PyObjectRef) -> PyResult<Message> {
    if let Some(py_msg) = obj.downcast_ref::<PyMessage>() {
        return Ok(py_msg.to_message());
    }
    encode_python_payload(vm, obj)
}

impl PyZeroCopyDescriptor {
    pub fn total_buffer_bytes(&self, vm: &VirtualMachine) -> usize {
        self.buffers
            .iter()
            .filter_map(|b| b.downcast_ref::<PyBytes>().map(|x| x.as_bytes().len()))
            .sum()
    }

    pub fn to_header(&self, _vm: &VirtualMachine) -> ZeroCopyDescriptorHeader {
        ZeroCopyDescriptorHeader {
            version: self.version,
            buffer_count: self.buffers.len(),
            buffer_lengths: self
                .buffers
                .iter()
                .filter_map(|b| b.downcast_ref::<PyBytes>().map(|x| x.as_bytes().len()))
                .collect(),
            dtype: self.dtype.clone(),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            transport: self.transport.clone(),
            checksum: self.checksum.clone(),
        }
    }

    pub fn serialize_single(&self, vm: &VirtualMachine) -> PyResult<Vec<u8>> {
        let header = self.to_header(vm);
        let header_bytes = bincode::serialize(&header).map_err(|e| vm.new_value_error(e.to_string()))?;
        let header_len = header_bytes.len() as u32;
        let total_data: usize = header.buffer_lengths.iter().sum();
        let mut out = Vec::with_capacity(4 + header_bytes.len() + total_data);
        out.extend_from_slice(&header_len.to_le_bytes());
        out.extend_from_slice(&header_bytes);
        for buf in &self.buffers {
            let bytes = buf
                .downcast_ref::<PyBytes>()
                .ok_or_else(|| vm.new_type_error("buffer must be bytes-like"))?;
            out.extend_from_slice(bytes.as_bytes());
        }
        Ok(out)
    }

    pub fn raw_buffer_data(&self, vm: &VirtualMachine) -> PyResult<Vec<Vec<u8>>> {
        self.buffers
            .iter()
            .map(|b| {
                b.downcast_ref::<PyBytes>()
                    .map(|x| x.as_bytes().to_vec())
                    .ok_or_else(|| vm.new_type_error("buffer must be bytes-like"))
            })
            .collect()
    }

    pub fn from_wire(
        vm: &VirtualMachine,
        header: ZeroCopyDescriptorHeader,
        raw_buffers: Vec<Vec<u8>>,
    ) -> Self {
        Self {
            version: header.version,
            buffers: raw_buffers
                .into_iter()
                .map(|b| vm.ctx.new_bytes(b).into())
                .collect(),
            dtype: header.dtype,
            shape: header.shape,
            strides: header.strides,
            transport: header.transport,
            checksum: header.checksum,
        }
    }
}

pub fn ensure_contiguous_buffer(vm: &VirtualMachine, item: &PyObjectRef) -> PyResult<PyObjectRef> {
    if item.downcast_ref::<PyBytes>().is_some() {
        return Ok(item.clone());
    }
    let builtins = vm.import("builtins", 0)?;
    let bytes_fn = builtins.get_attr("bytes", vm)?;
    let bytes_obj = bytes_fn.call((item.clone(),), vm)?;
    if bytes_obj.downcast_ref::<PyBytes>().is_none() {
        return Err(vm.new_value_error(
            "ZeroCopyDescriptor.buffers items must expose a contiguous Python buffer",
        ));
    }
    Ok(bytes_obj)
}
