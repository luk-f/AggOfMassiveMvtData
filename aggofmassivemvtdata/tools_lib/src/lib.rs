use pyo3::prelude::*;
use rayon::prelude::*;
use pyo3::wrap_pyfunction;
use std::f64;
use std::time::SystemTime;

const EARTH_RADIUS_IN_METERS: f64 = 6372797.560856;

fn to_radians(val: &f64) -> f64 {
    val * f64::consts::PI / 180.
}

fn haversine(source: &(f64, f64), destination: &(f64, f64), radius: Option<f64>) -> f64 {
    // source and destination are coordinates (lat, lon)
    let radius = radius.unwrap_or(EARTH_RADIUS_IN_METERS);
    let lat_1_rad = to_radians(&source.0);
    let lon_1_rad = to_radians(&source.1);
    let lat_2_rad = to_radians(&destination.0);
    let lon_2_rad = to_radians(&destination.1);
    let lat_arc = &lat_2_rad - &lat_1_rad;
    let lon_arc = &lon_2_rad - &lon_1_rad;
    let sin_half_lat = (&lat_arc / 2.).sin();
    let sin_half_lon = (&lon_arc / 2.).sin();
    let prod_cos_lat = lat_1_rad.cos() * lat_2_rad.cos();
    let a = (sin_half_lat*sin_half_lat) + (prod_cos_lat * sin_half_lon*sin_half_lon);
    let c = 2. * a.sqrt().asin();
    return radius * c;
}

fn bulk_haversine(pairs: &Vec<((f64, f64), (f64, f64))>, radius: Option<f64>) -> Vec<f64> {
    // pairs: [[[source_1_lat, source_1_lon], [dest_1_lat, dest_1_lon]], [source_2_lat, source_2_lon], [dest_2_lat, dest_2_lon]], ...]
    let radius = radius.unwrap_or(EARTH_RADIUS_IN_METERS);
    let distances = pairs.into_par_iter().map(|pair| {
        let source = &pair.0;
        let destination = &pair.1;
        haversine(source, destination, Some(radius))
    }).collect();
    return distances;
}

#[pymodule]
fn tools_lib(_py: Python, m: &PyModule) -> PyResult<()> {
    #[pyfn(m, "haversine")]
    #[text_signature = "(source, destination, radius=6372797.560856)"]
    fn haversine_py(_py: Python, source: (f64, f64), destination: (f64, f64), radius: Option<f64>) -> PyResult<f64>{
        Ok(haversine(&source, &destination, radius))
    }
    #[pyfn(m, "bulk_haversine")]
    #[text_signature = "(pairs, radius=6372797.560856)"]
    fn bulk_haversine_py(_py: Python, pairs: Vec<((f64, f64), (f64, f64))>, radius: Option<f64>) -> PyResult<Vec<f64>> {
        Ok(bulk_haversine(&pairs, radius))
    }
    Ok(())
}
