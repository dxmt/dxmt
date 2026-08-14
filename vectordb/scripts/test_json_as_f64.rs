// Test serde_json::Value::as_f64() with integers
use serde_json::json;

fn main() {
    // Test with integer
    let int_value = json!(2015);
    println!("int_value: {:?}", int_value);
    println!("int_value.as_f64(): {:?}", int_value.as_f64());
    println!("int_value.as_i64(): {:?}", int_value.as_i64());
    println!("int_value.as_u64(): {:?}", int_value.as_u64());

    // Test with float
    let float_value = json!(85.5);
    println!("\nfloat_value: {:?}", float_value);
    println!("float_value.as_f64(): {:?}", float_value.as_f64());

    // Test comparison
    let year = json!(2015);
    let threshold = json!(2000);

    println!("\nyear: {:?}, threshold: {:?}", year, threshold);

    if let (Some(v), Some(t)) = (year.as_f64(), threshold.as_f64()) {
        println!("v: {}, t: {}", v, t);
        println!("v >= t: {}", v >= t);
    } else {
        println!("Failed to convert to f64!");
    }

    // Test with object
    let obj = json!({"year": 2015, "popularity": 85});
    if let Some(map) = obj.as_object() {
        if let Some(year_val) = map.get("year") {
            println!("\nExtracted year from object: {:?}", year_val);
            println!("year.as_f64(): {:?}", year_val.as_f64());
        }
    }
}
