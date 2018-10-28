#[macro_use]
extern crate frames;

#[derive(Frame)]
struct Me {
    name: &'static str,
    age: i32
}

fn main() {

    // This is how we would *like* things to work

    // define_frame!(MyFrame, name: &'static str, age: i32);

    let mut frame: me::MeFrame = frame![
        ["alex", 32],
        ["bob",  42],
        [NA,    123]
    ];

    // frame.insert(Me { name: "more", age: 84 });
    // frame.insert(("gnamer", 432));
    frame.insert_row((Some("gnomer"), None));
    println!("{:?}", frame);

    // let other: Frame3<i32,

    // frame.join
}
