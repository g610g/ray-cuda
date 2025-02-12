use bio::io::fastq;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

//creates python bindings that will be used for parsing fastq files
#[pyfunction]
fn parse_fastq_file(file_path: String) -> PyResult<Vec<String>> {
    match fastq::Reader::from_file(file_path) {
        Ok(reader) => {
            let mut read_vector = vec![];
            for result in reader.records() {
                if let Ok(record) = result {
                    let string_record = byte_to_string(record.seq());
                    read_vector.push(string_record);
                }
            }
            return Ok(read_vector);
        }
        Err(e) => {
            //code panics for now. Will add better error handling
            panic!("Something went wrong during finding the file")
        }
    }

    Err(PyErr::new::<PyTypeError, _>("Something went wrong"))
}

//this function unwraps for now. Add better error handling
fn byte_to_string(byte_array: &[u8]) -> String {
    std::str::from_utf8(byte_array).unwrap().to_string()
}

#[pymodule]
fn fastq_parser(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_function(wrap_pyfunction!(parse_fastq_file, m)?)?;
    Ok(())
}
