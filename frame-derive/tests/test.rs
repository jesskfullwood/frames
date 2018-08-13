#[macro_use]
extern crate frame_derive;
extern crate frames;
extern crate frunk;
#[macro_use]
extern crate frunk_derives;
#[macro_use]
extern crate frunk_core;

#[derive(Clone, Debug, Frame)]
struct Policy {
    id: u32,
    price: f64,
    age: u32,
    holder_id: u32,
    pol_type: String
}

#[derive(Clone, Debug, Frame, Generic)]
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
fn test_frunky() {
    let person = Person {
        id: 101,
        sex: Sex::Male,
        name: "Barold".into(),
        age: 32
    };

    let h = hlist![10, Sex::Male, String::from("hello"), 123];
    type Parts = (u32, Sex, String, u32);
    let parts: Parts = frunk::from_generic(h.clone());
    let p1: Person = frunk::from_generic(h);
    let p2: Person = frunk::convert_from(parts.clone());
}

#[test]
fn test_derive() {
    let people = [
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
            age: 33
        },
    ];

    let peopleframe = PersonFrame::from_rows(people);



}

// type PersonFrame = Frame4<Id, Sex, Name, Age>
