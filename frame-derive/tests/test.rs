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

#[derive(Copy, Clone, Debug)]
enum Sex {
    Male,
    Female
}

#[test]
fn test_derive() {
    let people = vec![
        Person {
            id: 101,
            sex: Sex::Male,
            name: "Barold".into(),
            age: 32
        },
        Person {
            id: 102,
            sex: Sex::Female,
            name: "Baretta".into(),
            age: 23
        }];

    let frame = person::PersonFrame::from(people);
}

// type PersonFrame = Frame4<Id, Sex, Name, Age>
