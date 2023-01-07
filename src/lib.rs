use crossbeam::channel::bounded;
use fasttext::FastText;
use log::{debug, error};
use ndarray::{Array2, Ix2};
use numpy::ToPyArray;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString};
use rayon::prelude::*;
use std::collections::BTreeMap;

const CHANNEL_SIZE: usize = 128;

#[pyclass(name = "FastText")]
struct FastTextPy(FastText);

/// load model from path.
#[pyfunction]
#[pyo3(text_signature = "(path)")]
fn load_model(path: &str) -> PyResult<FastTextPy> {
    let mut model = FastText::new();
    if let Err(e) = model.load_model(path) {
        Err(PyException::new_err(e))
    } else {
        debug!("model loaded");
        Ok(FastTextPy(model))
    }
}

#[pymethods]
impl FastTextPy {
    /// batch texts prediction using multithreading.
    ///
    /// Args:
    ///     texts: a list of strings
    ///     label_to_int: a mapping from fasttext label to a positive i16
    ///     k: output k predictions per text
    ///     threshold: the minimal accuracy
    ///
    /// Returns:
    ///     A label, probability pairs in np.ndarray(i16) and np.ndarray(f32)
    ///     format. Where `-1` is used to represent label not found in label_to_int
    #[pyo3(text_signature = "($self, texts, label_to_int, k, threshold)")]
    #[args(k = "1", threshold = "-1.0")]
    fn batch(
        &self,
        texts: PyObject,
        label_to_int: &PyDict,
        k: i32,
        threshold: f32,
        py: Python,
    ) -> PyResult<(PyObject, PyObject)> {
        let counts = texts.as_ref(py).downcast::<PyList>()?.len();
        let label_dict: BTreeMap<String, i16> = label_to_int.extract()?;
        let mut labels = Array2::<i16>::default(Ix2(counts, k as usize));
        let mut probs = Array2::<f32>::default(Ix2(counts, k as usize));
        let (text_sender, text_receiver) = bounded::<Option<String>>(CHANNEL_SIZE);
        let (result_sender, result_receiver) = bounded(CHANNEL_SIZE);
        py.allow_threads(|| {
            std::thread::scope(|s| {
                // text sender
                s.spawn(|| {
                    Python::with_gil(|py| {
                        let texts_iter =
                            texts.as_ref(py)
                                .downcast::<PyList>()
                                .unwrap()
                                .iter()
                                .map(|s| {
                                    s.downcast::<PyString>()
                                        .ok()
                                        .map(|s| match s.to_str() {
                                            Ok(s) => Some(s.to_string()),
                                            Err(e) => {
                                                py.allow_threads(|| {
                                                    error!("Non-string element encountered in input, ignoring: {e}");
                                                });
                                                None
                                            }
                                        })
                                        .flatten()
                                });
                        for text in texts_iter {
                            if py.allow_threads(|| {
                                debug!("text sent: {:?}", text);
                                 text_sender.send(text)
                            }).is_err() {
                                break;
                            };
                        }
                        drop(text_sender);
                    });
                    debug!("text sender thread finished");
                });
                // processor
                s.spawn(|| {
                    text_receiver
                        .iter()
                        .enumerate()
                        .par_bridge()
                        .map(|(i, s)| {
                            let result = if let Some(s) = s {
                                debug!("text received: {:?}", s);
                                match self.0.predict(&s, k, threshold) {
                                    Ok(predictions) => predictions
                                        .into_iter()
                                        .map(|p| (*label_dict.get(&p.label).unwrap_or(&-1), p.prob))
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
                    debug!("processor thread finished");
                });
                // result writer
                s.spawn(|| {
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
}

#[pymodule]
fn fasttext_parallel(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    pyo3_log::init();
    m.add_function(wrap_pyfunction!(load_model, m)?)?;
    m.add_class::<FastTextPy>()?;
    Ok(())
}
