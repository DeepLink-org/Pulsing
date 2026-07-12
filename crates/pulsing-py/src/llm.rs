//! PyO3 bindings for `pulsing-forge` LLM client.

use pulsing_forge::llm::{LlmClient, LlmError, LlmMessage, LlmStream, StreamRequest};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use serde_json::Value;

use crate::forge::{block_on_tool, json_to_py, py_to_json};

#[pyclass(name = "LlmClient")]
#[derive(Clone)]
pub struct PyLlmClient {
    inner: LlmClient,
}

#[pyclass(name = "LlmStream")]
pub struct PyLlmStream {
    chunks: Vec<String>,
    final_message: LlmMessage,
    entered: bool,
}

#[pymethods]
impl PyLlmClient {
    #[new]
    #[pyo3(signature = (*, provider="anthropic", api_key=None, base_url=None))]
    fn new(provider: &str, api_key: Option<String>, base_url: Option<String>) -> PyResult<Self> {
        let inner = LlmClient::new(provider, api_key, base_url)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    #[getter]
    fn provider(&self) -> String {
        self.inner.provider().as_str().to_string()
    }

    #[pyo3(signature = (*, model, max_tokens, messages, system=None, tools=None))]
    fn stream_messages(
        &self,
        py: Python<'_>,
        model: String,
        max_tokens: u32,
        messages: &Bound<'_, PyAny>,
        system: Option<String>,
        tools: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyLlmStream> {
        let messages_json = py_list_to_values(py, messages)?;
        let tools_json = match tools {
            Some(t) => py_list_to_values(py, t)?,
            None => Vec::new(),
        };
        let req = StreamRequest {
            model,
            max_tokens,
            messages: messages_json,
            system,
            tools: tools_json,
        };
        let client = self.inner.clone();
        let stream = block_on_tool(async move { client.stream_messages(req).await })
            .map_err(|e: LlmError| PyRuntimeError::new_err(e.to_string()))?;
        Ok(PyLlmStream::from_rust(stream))
    }

    #[staticmethod]
    fn error_message(exc: &Bound<'_, PyAny>) -> String {
        exc.str().ok().map(|s| s.to_string()).unwrap_or_default()
    }

    fn is_authentication_error(&self, exc: &Bound<'_, PyAny>) -> PyResult<bool> {
        Ok(classify_py_error(exc)?.is_authentication_error())
    }

    fn is_retryable_error(&self, exc: &Bound<'_, PyAny>) -> PyResult<bool> {
        Ok(classify_py_error(exc)?.is_retryable_error())
    }

    fn is_api_error(&self, exc: &Bound<'_, PyAny>) -> PyResult<bool> {
        Ok(classify_py_error(exc)?.is_api_error())
    }
}

#[pymethods]
impl PyLlmStream {
    fn __enter__(mut slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf.entered = true;
        slf
    }

    fn __exit__(
        &mut self,
        _exc_type: Option<&Bound<'_, PyAny>>,
        _exc: Option<&Bound<'_, PyAny>>,
        _tb: Option<&Bound<'_, PyAny>>,
    ) -> bool {
        false
    }

    fn close(&self) {}

    #[getter]
    fn text_stream(slf: PyRef<'_, Self>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let iter = PyLlmTextIter {
            chunks: slf.chunks.clone(),
            index: 0,
        };
        Ok(Py::new(py, iter)?.into_any())
    }

    fn get_final_message(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        llm_message_to_py(py, &self.final_message)
    }
}

#[pyclass]
struct PyLlmTextIter {
    chunks: Vec<String>,
    index: usize,
}

#[pymethods]
impl PyLlmTextIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> PyResult<Option<String>> {
        if slf.index >= slf.chunks.len() {
            return Ok(None);
        }
        let chunk = slf.chunks[slf.index].clone();
        slf.index += 1;
        Ok(Some(chunk))
    }
}

impl PyLlmStream {
    fn from_rust(stream: LlmStream) -> Self {
        let chunks = stream.text_chunks();
        let final_message = stream.final_message();
        Self {
            chunks,
            final_message,
            entered: false,
        }
    }
}

fn py_list_to_values(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Vec<Value>> {
    if let Ok(list) = obj.downcast::<PyList>() {
        list.iter()
            .map(|item| py_to_json(&item))
            .collect::<PyResult<Vec<_>>>()
    } else {
        Ok(vec![py_to_json(obj)?])
    }
}

fn llm_message_to_py(py: Python<'_>, msg: &LlmMessage) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    dict.set_item(
        "content",
        json_to_py(py, &Value::Array(msg.content.clone()))?,
    )?;
    if let Some(usage) = &msg.usage {
        let u = PyDict::new(py);
        u.set_item("input_tokens", usage.input_tokens)?;
        u.set_item("output_tokens", usage.output_tokens)?;
        dict.set_item("usage", u)?;
    } else {
        dict.set_item("usage", py.None())?;
    }
    dict.set_item("stop_reason", msg.stop_reason.clone())?;
    Ok(dict.into())
}

fn classify_py_error(exc: &Bound<'_, PyAny>) -> PyResult<LlmError> {
    if let Ok(status) = exc.getattr("status_code") {
        if let Ok(code) = status.extract::<u16>() {
            let body = exc.str().map(|s| s.to_string()).unwrap_or_default();
            return Ok(LlmError::Api { status: code, body });
        }
    }
    let msg = exc.str().map(|s| s.to_string()).unwrap_or_default();
    if msg.contains("401") || msg.contains("403") || msg.to_lowercase().contains("auth") {
        return Ok(LlmError::Api {
            status: 401,
            body: msg,
        });
    }
    if msg.to_lowercase().contains("rate limit") {
        return Ok(LlmError::Api {
            status: 429,
            body: msg,
        });
    }
    Ok(LlmError::Other(msg))
}

pub fn add_to_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyLlmClient>()?;
    m.add_class::<PyLlmStream>()?;
    Ok(())
}
