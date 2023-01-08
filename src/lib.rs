use crossbeam::channel::{bounded, Receiver, Sender};
use fasttext::FastText;
use log::{debug, error};
use ndarray::{Array2, Ix2};
use numpy::ToPyArray;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyDict, PyList, PyString};
use rayon::prelude::*;
use std::cmp::max;
use std::collections::BTreeMap;
use std::thread::available_parallelism;

const CHANNEL_SIZE: usize = 128;
const MIN_THREADS: usize = 3;

#[pyclass(name = "FastText")]
struct FastTextPy {
    model: FastText,
    label_dict: BTreeMap<String, i16>,
    reverse_label_dict: BTreeMap<i16, String>,
}

/// load model from path.
///
/// Args:
///     path: file path of the model
///     label_to_int: a mapping from fasttext label to a positive i16
#[pyfunction]
#[pyo3(text_signature = "(path)")]
fn load_model(path: &str) -> PyResult<FastTextPy> {
    let mut model = FastText::new();
    if let Err(e) = model.load_model(path) {
        Err(PyException::new_err(e))
    } else {
        debug!("model loaded");
        let labels = match model.get_labels() {
            Ok((labels, _)) => labels,
            Err(e) => return Err(PyException::new_err(e)),
        };
        let label_dict: BTreeMap<String, i16> = labels
            .iter()
            .into_iter()
            .enumerate()
            .map(|(i, lab)| (lab.clone(), i as i16))
            .collect();
        let reverse_label_dict: BTreeMap<i16, String> = labels
            .into_iter()
            .enumerate()
            .map(|(i, lab)| (i as i16, lab))
            .collect();
        Ok(FastTextPy {
            model,
            label_dict,
            reverse_label_dict,
        })
    }
}

#[pymethods]
impl FastTextPy {
    /// batch texts prediction using multithreading.
    ///
    /// Args:
    ///     texts: a list of strings
    ///     k: output k predictions per text
    ///     threshold: the minimal accuracy
    ///
    /// Returns:
    ///     A label, probability pairs in np.ndarray(i16) and np.ndarray(f32)
    ///     format. Where `-1` is used to represent label not found in label_to_int
    #[pyo3(text_signature = "($self, texts, k, threshold)")]
    #[args(k = "1", threshold = "-1.0")]
    fn batch(
        &self,
        texts: PyObject,
        k: i32,
        threshold: f32,
        py: Python,
    ) -> PyResult<(PyObject, PyObject)> {
        let counts = texts.as_ref(py).downcast::<PyList>()?.len();
        let mut labels = Array2::<i16>::default(Ix2(counts, k as usize));
        let mut probs = Array2::<f32>::default(Ix2(counts, k as usize));
        let (text_sender, text_receiver) = bounded::<Option<String>>(CHANNEL_SIZE);
        let (result_sender, result_receiver) = bounded(CHANNEL_SIZE);
        py.allow_threads(|| {
            rayon::scope(|s| {
                // text sender
                s.spawn(|_| {
                    Python::with_gil(|py| {
                        let texts = texts.as_ref(py).downcast::<PyList>().unwrap();
                        send_text(texts, text_sender, py);
                    });
                    debug!("text sender thread finished");
                });

                // processor
                s.spawn(|_| {
                    predict_test(self, text_receiver, result_sender, k, threshold);
                    debug!("processor thread finished");
                });

                // result writer
                s.spawn(|_| {
                    for (i, result) in result_receiver {
                        debug!("result {i} received");
                        let mut label = labels.row_mut(i);
                        let mut prob = probs.row_mut(i);
                        label.as_slice_mut().unwrap()[..result.0.len()].copy_from_slice(&result.0);
                        prob.as_slice_mut().unwrap()[..result.1.len()].copy_from_slice(&result.1);
                    }
                });
            });
        });
        let labels = Python::with_gil(|py| labels.to_pyarray(py).to_object(py));
        let probs = Python::with_gil(|py| probs.to_pyarray(py).to_object(py));
        Ok((labels, probs))
    }

    /// get the mapping from label index to label.
    ///
    /// Returns:
    ///     A dictionary mapping from integer to labels.
    #[pyo3(text_signature = "($self)")]
    fn get_labels<'a>(&self, py: Python<'a>) -> &'a PyDict {
        self.reverse_label_dict.clone().into_py_dict(py)
    }

    /// get a label by the id
    ///
    /// Args:
    ///     id: the id of a label
    ///
    /// Returns:
    ///     the label corresponding to the given id.
    #[pyo3(text_signature = "($self, id)")]
    fn get_label_by_id(&self, id: i16) -> Option<&String> {
        self.reverse_label_dict.get(&id)
    }
}

#[inline]
fn send_text(texts: &PyList, text_sender: Sender<Option<String>>, py: Python) {
    let texts_iter = texts.iter().map(|s| {
        s.downcast::<PyString>()
            .ok()
            .and_then(|s| match s.to_str() {
                Ok(s) => Some(s.to_string()),
                Err(e) => {
                    py.allow_threads(|| {
                        error!("Non-string element encountered in input, ignoring: {e}");
                    });
                    None
                }
            })
    });
    for text in texts_iter {
        let send_result = py.allow_threads(|| {
            debug!("text sent: {:?}", text);
            text_sender.send(text)
        });
        if send_result.is_err() {
            break;
        };
    }
    drop(text_sender);
}

type ResultSender = Sender<(usize, (Vec<i16>, Vec<f32>))>;

#[inline]
fn predict_test(
    model: &FastTextPy,
    text_receiver: Receiver<Option<String>>,
    result_sender: ResultSender,
    k: i32,
    threshold: f32,
) {
    text_receiver
        .iter()
        .enumerate()
        .par_bridge()
        .map(|(i, s)| {
            let result = if let Some(s) = s {
                debug!("text received: {:?}", s);
                match model.model.predict(&s, k, threshold) {
                    Ok(predictions) => predictions
                        .into_iter()
                        .map(|p| (model.label_dict.get(&p.label).unwrap_or(&-1), p.prob))
                        .unzip(),
                    Err(e) => {
                        error!("Error making prediction, ignoring: {e}");
                        (vec![], vec![])
                    }
                }
            } else {
                (vec![], vec![])
            };
            if result_sender.send((i, result)).is_err() {
                None
            } else {
                Some(())
            }
        })
        .while_some()
        .for_each(|_| {});
    drop(result_sender);
}

#[pymodule]
fn fasttext_parallel(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    pyo3_log::init();
    let num_parallelism = available_parallelism()
        .map_err(|e| PyException::new_err(format!("failed to initialize rayon crate, {e}")))?;
    rayon::ThreadPoolBuilder::new()
        .num_threads(max(MIN_THREADS, num_parallelism.get()))
        .build_global()
        .map_err(|e| PyException::new_err(format!("failed to initialize rayon crate, {e}")))?;
    m.add_function(wrap_pyfunction!(load_model, m)?)?;
    m.add_class::<FastTextPy>()?;
    Ok(())
}
