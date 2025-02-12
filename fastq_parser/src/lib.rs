use bio::io::fastq;
use pyo3::prelude::*;

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}
#[pyfunction]
fn parse_fastq_file(file_path: String) -> PyResult<Vec<String>> {
    let mut read_vector = vec![];
    let mut reader = fastq::Reader::from_file(file_path).unwrap();
    let mut nb_read = 0;
    let mut nb_bases = 0;
    for result in reader.records() {
        let record = result.unwrap();
        let string_record = std::str::from_utf8(record.seq()).unwrap().to_string();
        read_vector.push(string_record);
    }

    Ok(read_vector)
}
#[pymodule]
fn fastq_parser(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_function(wrap_pyfunction!(parse_fastq_file, m)?)?;
    Ok(())
}
