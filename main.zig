const std = @import("std");
const net = @import("Network.zig");
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;
const RndGen = std.rand.DefaultPrng;
const simd = std.simd;
const math = std.math;
const Network = net.Network;

pub inline fn annealLearningRate(epoc: usize) f32 {
    return 1.0/math.log(f32, math.e, @as(f32,@floatFromInt(epoc))+math.e-1.0);
}

pub const modules = .{

};


const features :usize = 28*28; //NUM_INPUTS
const catagories :usize = 10; //NUM_OUTPUTS

// pub fn main() !void {
// //     try app.init();
// //     for(0..100) |_| {
//         app.update();
//         std.time.sleep(100);
//     }
//     app.deinit();
// }

pub fn main() !void {

    var mallocDaddy = std.heap.GeneralPurposeAllocator(.{}){};
    const malloc = mallocDaddy.allocator();

    const timeStart = try std.time.Instant.now();

    const VecLength :u32 = if(std.simd.suggestVectorLength(f32)) |val| val else 4;

    std.debug.print("\nMax Vec Length for SIMD: {any}\n", .{VecLength});

    var inputs = std.ArrayList([]f32).init(malloc);
    defer inputs.deinit();

    var targets = std.ArrayList([]f32).init(malloc);
    defer targets.deinit();


    var file = try std.fs.cwd().openFile("mnist_train.csv", .{});
    //NOTE :: no deference of closing. I close manually later.

    var buf_reader = std.io.bufferedReader(file.reader());
    var in_stream = buf_reader.reader();

    var line = std.ArrayList(u8).init(malloc);
    defer(line.deinit());

    const writer = line.writer();

    var line_no: usize = 0;
    while(in_stream.streamUntilDelimiter(writer, '\n', null)) : (line_no += 1) {
        defer line.clearRetainingCapacity();
        if(line_no > 0){
            const textID = line.items[0]-48;
            var answer = try malloc.alloc(f32, 10);
            @memset(answer, 0.0);
            answer[textID] = 1.0;

            try targets.append(answer);

            const input = try malloc.alloc(f32, features);

            var lineSplit = std.mem.split(u8, line.items[1..], ",");
            var counter:usize = 0;
            var avgInput:f32 = 0.0;

            while(lineSplit.next()) |x| {
                if(std.fmt.parseInt(u32, x, 10)) |val| {
                    input[counter] = (@as(f32, (@floatFromInt(val))));
                    avgInput += input[counter];
                    counter += 1;
                } else |_| {}
            }
            avgInput /= @as(f32, @floatFromInt(input.len));
            var secondAvg:f32 = 0.0;
            for(0 .. input.len) |a| {
                input[a] = (input[a] - avgInput);
                secondAvg += @abs(input[a]);
                //input[a] = if(input[a] <= 5.0) 0.0 else 1.0;
            }
            secondAvg /= @as(f32, @floatFromInt(input.len));
            for(0 .. input.len) |a| {
                input[a] /= secondAvg;
            }
            try inputs.append(input);
        }
    } else |err| switch (err) {
        error.EndOfStream => {},
        else => return err,
    }

    file.close();

    var myNet :Network = try Network.init(malloc);

    var epoc :usize = 1;
    var numWrong :usize = 100000;
    var totalWrong:usize = 0;
    var breached :bool = false;

    const dataReadIn = inputs.items.len;

    while (numWrong > dataReadIn / 50 and epoc < 205) {
        std.debug.print("=======\nFull Training This Epoch: {}\n", .{breached});
        numWrong = 0;
        const learnRate = annealLearningRate(epoc);
        for(0 .. inputs.items.len) |i| {
            const input = inputs.items[i];
            const results  = try myNet.feed(input);
            const target = targets.items[i];
            if(breached) {
                _ = try myNet.train(target);
                if(i % 250 == 0) {
                    myNet.update(learnRate);
                }
            }

            if(std.mem.indexOfMax(f32, results) != std.mem.indexOfMax(f32, target)) {
                numWrong += 1;
                totalWrong += 1;
                if(!breached) {
                    _ = try myNet.train(target);
                    if(totalWrong % 25 == 0) {
                        myNet.update(learnRate);
                    }
                }
            }
            if(i % 600 == 0) {
                std.debug.print(".", .{});
            }

        }
        //if(!breached and totalWrong%5>3) {
        //    net.update();
        //}
        breached = (numWrong < dataReadIn / 6);
        std.debug.print("\n{}::) Num wrong this epoch: {}\n", .{epoc, numWrong});
        epoc += 1;
    }

    file = try std.fs.cwd().openFile("mnist_test.csv", .{});
    defer (file.close());

    buf_reader = std.io.bufferedReader(file.reader());
    in_stream = buf_reader.reader();

    inputs.clearAndFree();
    targets.clearAndFree();

    line_no = 0;
    while(in_stream.streamUntilDelimiter(writer, '\n', null)) : (line_no += 1) {
        defer line.clearRetainingCapacity();
        if(line_no > 0){
            const textID = line.items[0]-48;
            var answer = try malloc.alloc(f32, 10);
            @memset(answer, 0.0);
            answer[textID] = 1.0;

            try targets.append(answer);

            const input = try malloc.alloc(f32, features);

            var lineSplit = std.mem.split(u8, line.items[1..], ",");
            var counter:usize = 0;
            var avgInput:f32 = 0.0;

            while(lineSplit.next()) |x| {
                if(std.fmt.parseInt(u32, x, 10)) |val| {
                    input[counter] = (@as(f32, (@floatFromInt(val))));
                    avgInput += input[counter];
                    counter += 1;
                } else |_| {}
            }
            avgInput /= @as(f32, @floatFromInt(input.len));
            var secondAvg:f32 = 0.0;
            for(0 .. input.len) |a| {
                input[a] = (input[a] - avgInput);
                secondAvg += @abs(input[a]);
                //input[a] = if(input[a] <= 5.0) 0.0 else 1.0;
            }
            secondAvg /= @as(f32, @floatFromInt(input.len));
            for(0 .. input.len) |a| {
                input[a] /= secondAvg;
            }
            try inputs.append(input);
        }
    } else |err| switch (err) {
        error.EndOfStream => {},
        else => return err,
    }

    std.debug.print("Beginning Testing\n", .{});

    myNet.releaseMode();
    numWrong = 0;
    for(0 .. inputs.items.len) |i| {
        const results = try myNet.feed(inputs.items[i]);
        if(std.mem.indexOfMax(f32, results) != std.mem.indexOfMax(f32, targets.items[i])) {
            numWrong += 1;
        }
        if(i % 100 == 0) {
            std.debug.print(".", .{});
        }
        //std.debug.print("{}::\n{any}\nvs\n    {any}\n", .{i, results, targets.items[i]});
    }

    std.debug.print("\nNum Wrong on testing set: {}\n", .{numWrong});

    inputs.clearAndFree();
    targets.clearAndFree();
    myNet.deinit();

    const timeEnd = try std.time.Instant.now();

    std.debug.print("Elapse time (init->train/test->free):\n{any} ms\n", .{timeEnd.since(timeStart)/1000000});

}
