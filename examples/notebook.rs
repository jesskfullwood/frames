#[macro_use]
extern crate frames;

#[derive(Frame)]
struct Me {
    name: &'static str,
    age: i32
}

fn receive(frame: &me::MeFrame) {
    let namecol = frame.get(me::Name);
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

    receive(&frame);

    define_col!(Hi, &'static str);
    let frame = frame.add::<Hi, _>(col![NA, NA, NA, "hi"]).unwrap();
    let x: i32 = frame.get(me::Age).iter().sum();

    // let other: Frame3<i32,

    // frame.join
}
