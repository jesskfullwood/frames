#[macro_use]
extern crate frame_derive;
extern crate frames;

#[derive(Clone, Debug, Frame)]
struct Policy {
    id: u32,
    price: f64,
    age: u32,
    holder_id: u32,
    pol_type: String
}

#[derive(Clone, Debug, Frame)]
struct Person {
    id: u32,
    sex: Sex,
    name: String,
    age: u32,
}

#[derive(Clone, Debug)]
enum Sex {
    Male,
    Female
}

#[test]
fn test_derive() {
    unimplemented!()

}

// type PersonFrame = Frame4<Id, Sex, Name, Age>
