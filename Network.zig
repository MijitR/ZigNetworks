const std = @import("std");
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;
const RndGen = std.rand.DefaultPrng;
const simd = std.simd;
const math = std.math;

/////////////////////////////////////////////////
//SEMI-EDITABLE DATA: HYPER-PARAMS
const ETA :f32 = 0.001; //Learning Rate
const MOMENTUM :f32 = 0.9618; //Decelerate or push if necessary
const RMS_DECAY :f32 = 0.999618; //Squish if overexcited consistently
const WEIGHT_DECAY :f32 = 1.0;//0.9999618; //No citadels, knock 'em all down

const features :usize = 28*28; //NUM_INPUTS
const catagories :usize = 10; //NUM_OUTPUTS

const layerSizes:usize = 16*6; //SIZE OF ALL HIDDEN LAYERS
const numLayers:usize = 3; //NUMBER OF HIDDEN LAYERS

const dropoutChance: f32 = 0.0; //Applies only to hidden layers
const gradientClip: f32 = 61.8; //Don't overreact

pub inline fn annealLearningRate(epoc: usize) f32 {
    return 1.0/math.log(f32, math.e, @as(f32,@floatFromInt(epoc))+math.e-1.0);
}

pub fn clipGradient(gradient: f32) f32 {
    return std.math.clamp(gradient, -gradientClip, gradientClip);//@max(-gradientClip, @min(gradientClip, gradient));
}
/////////////////////////////////////////////////

pub const Network = struct {
    malloc: Allocator,
    page: []f32,
    inputLayer: InLayer,
    layers: []Layer,
    lastOutput: []f32,
    lastInfluence: []f32,
    neuronCount: usize,
    inputParams: usize,
    age: usize,
    dropout: bool,
    pub fn feed(self: *Network, input:[] f32) ![]f32 {
        var inStart : usize = 0;
        var inEnd:usize = features;
        var outStart: usize = features;
        var outEnd: usize = outStart + layerSizes;

        self.inputLayer.transfer(input[inStart..inEnd], self.page[outStart..outEnd], self.dropout);
        outStart = outEnd;
        inStart = inEnd;
        inEnd = inStart + layerSizes;

        inline for(0 .. (numLayers - 1)) |i| {
            outEnd = outStart + layerSizes;
            self.layers[i].transfer(self.page[inStart..inEnd], self.page[outStart..outEnd], self.dropout);
            outStart = outEnd;
            inStart = inEnd;
            inEnd = inStart + layerSizes;
        }

        outEnd += catagories;

        self.layers[numLayers-1].transfer(self.page[inStart..inEnd], self.page[outStart..outEnd], false);

        @memcpy(self.lastOutput, self.page[outEnd-catagories..outEnd]);

        return self.lastOutput;
    }
    pub fn train(self: *Network, targets: []f32) ![]f32 {
//         @setFloatMode(std.builtin.FloatMode.optimized);
        assert(targets.len == self.lastOutput.len);

        @memset(self.page, 0.0);

        inline for(0 .. catagories) |i| {

            self.page[self.inputParams + i] =

                (-targets[i]/(self.lastOutput[i]))
                    +
                ((1.0-targets[i])/(1.0-self.lastOutput[i]));

        }

        var inStart = self.inputParams - layerSizes;
        var inEnd = self.inputParams;
        var neurStart = self.inputParams;
        var neurEnd = self.inputParams + catagories;

        self.layers[numLayers-1].diffuse(self.page[neurStart..neurEnd], self.page[inStart..inEnd]);
        neurStart -= layerSizes;
        neurEnd -= catagories;
        inStart -= layerSizes;
        inEnd -= layerSizes;

        inline for(2 .. numLayers+1) |i| {
            self.layers[numLayers-i].diffuse(self.page[neurStart..neurEnd], self.page[inStart..inEnd]);
            neurStart -= layerSizes;
            neurEnd -= layerSizes;
            inStart -= layerSizes;
            inEnd -= layerSizes;
        }

        neurStart = features;
        neurEnd = features + layerSizes;
        inStart = 0;
        inEnd = features;

        self.inputLayer.diffuse(self.page[neurStart..neurEnd], self.page[inStart..inEnd]);

        @memcpy(self.lastInfluence, self.page[0..features]);

        self.age += 1;

        return self.lastInfluence;
    }
    pub fn update(self: *Network, rateMod: f32) void {
        if(self.age > 0) {
            self.inputLayer.update(rateMod);
            for(0 .. numLayers) |i| {
                self.layers[i].update(rateMod);
            } self.age = 0;
        }
    }
    pub fn init(myMalloc: Allocator) !Network {
        std.debug.print(
            "Creating Network-->\nFeatures: {} Catagories: {} (HiddenLayerSize X Count)({} x {})\nDropout: {}\n",
            .{features, catagories, layerSizes, numLayers, dropoutChance}
        );
        var myLayers: []Layer = try myMalloc.alloc(Layer, (numLayers - 1) + 1);
        errdefer myMalloc.free(myLayers);

        const myInfluence: []f32 = try myMalloc.alloc(f32, features);
        errdefer myMalloc.free(myInfluence);

        var neurCount: usize = catagories;
        var myInputParams: usize = features;

        const myInLayer :InLayer = try InLayer.init(features, layerSizes, numLayers+1, myMalloc);

        neurCount += layerSizes;
        myInputParams += layerSizes;

        for(0 .. numLayers-1) |i| {
            myLayers[i] = (try Layer.init(layerSizes, layerSizes, i, myMalloc));
            neurCount += layerSizes;
            myInputParams += layerSizes;
        }

        myLayers[numLayers-1] = try Layer.init(layerSizes, catagories, numLayers, myMalloc);

        const myLastOutput =  try myMalloc.alloc(f32, catagories);
        errdefer(myMalloc.free(myLastOutput));

        const myPage: []f32 = try myMalloc.alloc(f32, features + numLayers * layerSizes + catagories);
        errdefer(myMalloc.free(myPage));

        const myDropoutBool = dropoutChance > 0.0;

        const myAge :usize = 0;

        return Network {
            .malloc = myMalloc,
            .page = myPage,
            .inputLayer = myInLayer,
            .layers = myLayers,
            .lastOutput = myLastOutput,
            .lastInfluence = myInfluence,
            .neuronCount = neurCount,
            .inputParams = myInputParams,
            .age = myAge,
            .dropout = myDropoutBool,
        };
    }
    pub fn deinit(self: *Network) void {
        for(0 .. self.layers.len) |i| {
            self.layers[i].deinit();
        }
        self.malloc.free(self.layers);
        self.malloc.free(self.lastOutput);
        self.malloc.free(self.lastInfluence);
        self.malloc.free(self.page);
    }
    pub fn releaseMode(self: *Network) void {
        self.dropout = false;
    }
};

const InLayer = struct {
    malloc: Allocator,
    neurons: []InNeuron,
    fanIn: usize,
    pub fn transfer(self: *InLayer, input: []f32, result: []f32, dropout: bool) void {
        assert(result.len == self.neurons.len);
        assert(self.fanIn == input.len);
        const layerSize = self.neurons.len;
        for(0 .. layerSize) |i| {
            result[i] = self.neurons[i].consume(input, dropout);
        }
    }
    pub fn diffuse(self: *InLayer, errs: []f32, influence: []f32) void {
        assert(influence.len == self.fanIn);
        assert(errs.len == self.neurons.len);
        const layerSize = self.neurons.len;
        for(0 .. layerSize) |i| {
            self.neurons[i].emit(errs[i], influence);
        }
    }
    pub fn update(self :*InLayer, rateMod: f32) void {
        const layerSize = self.neurons.len;
        for(0 .. layerSize) |i| {
            self.neurons[i].update(rateMod);
        }
    }
    pub fn init(comptime inFeats: usize, size: usize, layerID: usize, myMalloc: Allocator) !InLayer {
        const myNeurons = try myMalloc.alloc(InNeuron, size);
        errdefer myMalloc.free(myNeurons);
        for(0 .. size) |i| {
            myNeurons[i] = try InNeuron.init(inFeats, layerID, layerID*size + i, myMalloc);
        }
        return InLayer {
            .malloc = myMalloc,
            .neurons = myNeurons,
            .fanIn = inFeats,
        };
    }
    pub fn deinit(self :*Layer) void {
        for(0 .. self.neurons.len) |i| {
            self.neurons[i].deinit();
        }
        self.malloc.free(self.neurons);
    }
};

const Layer = struct {
    malloc: Allocator,
    neurons: []StdNeuron,
    fanIn: usize,
    pub fn transfer(self: *Layer, input: []f32, result: []f32, dropout: bool) void {
        assert(result.len == self.neurons.len);
        assert(self.fanIn == input.len);
        const layerSize = self.neurons.len;
        for(0 .. layerSize) |i| {
            result[i] = self.neurons[i].consume(input, dropout);
        }
    }
    pub fn diffuse(self: *Layer, errs: []f32, influence: []f32) void {
        assert(influence.len == self.fanIn);
        assert(errs.len == self.neurons.len);
        const layerSize = self.neurons.len;
        for(0 .. layerSize) |i| {
            self.neurons[i].emit(errs[i], influence);
        }
    }
    pub fn update(self :*Layer, rateMod: f32) void {
        const layerSize = self.neurons.len;
        for(0 .. layerSize) |i| {
            self.neurons[i].update(rateMod);
        }
    }
    pub fn init(comptime inFeats: usize, size: usize, layerID: usize, myMalloc: Allocator) !Layer {
        const myNeurons = try myMalloc.alloc(StdNeuron, size);
        errdefer myMalloc.free(myNeurons);
        for(0 .. size) |i| {
            myNeurons[i] = try StdNeuron.init(inFeats, layerID, layerID*size + i, myMalloc);
        }
        return Layer {
            .malloc = myMalloc,
            .neurons = myNeurons,
            .fanIn = inFeats,
        };
    }
    pub fn deinit(self :*Layer) void {
        for(0 .. self.neurons.len) |i| {
            self.neurons[i].deinit();
        }
        self.malloc.free(self.neurons);
    }
};

const InNeuron = struct {
    comptime fanIn: usize = features,
    malloc: Allocator,
    weights: []f32,
    deltaWeights: []f32,
    weightVelocities: []f32,
    rmsProps: []f32,
    bias: f32,
    deltaBias: f32,
    biasVelocity: f32,
    biasRMS: f32,
    dwrtI: []f32,
    dwrtW: []f32,
    layerID: usize,
    gradient: f32,
    trainingInstances: f32,
    age: usize,
    rnd: std.Random.Xoshiro256,
    dropMag: f32,

    pub fn consume(self: *InNeuron, input: []f32, dropout: bool) f32 {
        assert(self.weights.len == input.len);
//         @setFloatMode(std.builtin.FloatMode.optimized);
        self.dropMag = 1.0;
        if(dropout) {
            assert(dropoutChance > 0.0);
            self.dropMag = 1.0/(1.0-dropoutChance);
            if(@abs(self.rnd.random().float(f32)) < dropoutChance) {
                self.gradient = 0.0;
                return 0.0;
            }
        }

        @memcpy(self.dwrtI, self.weights);
        @memcpy(self.dwrtW, input);

        //is it faster to do one loop instead of two memcpys??

        //for (0 .. input.len) |i| {
        //    self.dwrtI[i] = self.weights[i];
        //    self.dwrtW[i] = input[i];
        //}

        //probs not, I think intrinsics are utilized behind the scenes in memcpy

        const inVec: @Vector(self.fanIn, f32) = input[0..self.fanIn].*;
        const weightVec :@Vector(self.fanIn, f32) = self.weights[0..self.fanIn].*;

        const summer = inVec * weightVec;

        const sum = @reduce(.Add, summer) * self.dropMag + self.bias;

        //SNEAKY LEAKY RELU FIRST ACTIVATIONS
//         const res = @max(0.01*sum, sum);
//         self.gradient = res / sum;
        //SNEAKY RELU FIRST ACTIVATIONS
//          const res = @max(0, sum);
//          self.gradient = res / sum;
        //ELU FIRST ACTIVATIONS
        if(sum >= 0) {
            self.gradient = 1.0;
            return sum;
        }
        const res = @exp(sum) - 1.0;
        self.gradient = res + 1.0;
        //SWISH FIRST ACTIVATIONS
//            const exp = @exp(sum);
//            const sig = exp/(1.0+exp);
//            const res = sum * sig;
//            self.gradient = sig * (1.0+sum*(1.0-sig));
        //TANH FIRST ACTIVATIONS  -- NOTE: TANH behaves kind of slowly as a gatekeeper
//          const res = std.math.tanh(sum);
//          self.gradient = 1.0-res*res;

        return res;
    }
    pub fn emit(self: *InNeuron, err: f32, accum: []f32) void {
        if(self.gradient == 0.0) {
            self.trainingInstances += 1.0;
            return;
        }
//         @setFloatMode(std.builtin.FloatMode.optimized);
        var dist = err * self.gradient;

        self.deltaBias += dist;
        self.trainingInstances += 1.0;

        dist *= self.dropMag;

        const distSpray: @Vector(self.fanIn, f32) = @splat(dist);


        const inpVec: @Vector(self.fanIn, f32) = self.dwrtI[0..self.fanIn].*;
        const weiVec: @Vector(self.fanIn, f32) = self.dwrtW[0..self.fanIn].*;

        const accuVec: @Vector(self.fanIn, f32) = accum[0..self.fanIn].*;
        const deltaVec: @Vector(self.fanIn, f32) = self.deltaWeights[0..self.fanIn].*;

        const accuArray :[self.fanIn]f32 = @mulAdd(@Vector(self.fanIn, f32), distSpray, inpVec, accuVec);
        const deltaArray :[self.fanIn]f32 = @mulAdd(@Vector(self.fanIn, f32), distSpray, weiVec, deltaVec);

        @memcpy(accum, &accuArray);
        @memcpy(self.deltaWeights, &deltaArray);
    }
    pub fn update(self: *InNeuron, rateMod: f32) void {
//         @setFloatMode(std.builtin.FloatMode.optimized);
        assert(self.trainingInstances > 0);
        self.age += 1;
        const velScalar = 1.0 - std.math.pow(f32, MOMENTUM, @as(f32, @floatFromInt(self.age)));
        const rmsScalar = 1.0 - std.math.pow(f32, RMS_DECAY, @as(f32, @floatFromInt(self.age)));
        inline for(0 .. self.fanIn) |i| {
            const delta = self.deltaWeights[i] / self.trainingInstances;
            self.weightVelocities[i] = MOMENTUM * self.weightVelocities[i] + (1.0-MOMENTUM) * delta;
            self.rmsProps[i] = RMS_DECAY * self.rmsProps[i] + (1.0-RMS_DECAY) * delta * delta;
            self.weights[i] = self.weights[i] * WEIGHT_DECAY - ETA * rateMod *
                clipGradient((self.weightVelocities[i] / velScalar) / ((@sqrt(self.rmsProps[i]/rmsScalar) + 0.0001)));
        }
        @memset(self.deltaWeights, 0.0);
        const delta = self.deltaBias / self.trainingInstances;
        self.biasVelocity = MOMENTUM * self.biasVelocity + (1.0-MOMENTUM) * delta;
        self.biasRMS = RMS_DECAY * self.biasRMS + (1.0-RMS_DECAY) * delta * delta;
        self.bias = self.bias * WEIGHT_DECAY - ETA * rateMod *
            clipGradient((self.biasVelocity / velScalar) / ((@sqrt(self.biasRMS/rmsScalar) + 0.0001)));
        self.deltaBias = 0.0;
        self.trainingInstances = 0.0;
    }
    pub fn init(comptime inputSize: usize, myLayerID: usize, id: usize, myMalloc: Allocator) !InNeuron {
        const myWeights :[]f32 = try myMalloc.alloc(f32, inputSize);
        errdefer myMalloc.free(myWeights);
        const myDeltas :[]f32 = try myMalloc.alloc(f32, inputSize);
        errdefer myMalloc.free(myDeltas);
        const myDwrtI :[]f32 = try myMalloc.alloc(f32, inputSize);
        errdefer myMalloc.free(myDwrtI);
        const myDwrtW :[]f32 = try myMalloc.alloc(f32, inputSize);
        errdefer myMalloc.free(myDwrtW);
        const myVel :[]f32 = try myMalloc.alloc(f32, inputSize);
        errdefer myMalloc.free(myVel);
        const myRMSProps :[]f32 = try myMalloc.alloc(f32, inputSize);
        errdefer myMalloc.free(myRMSProps);

        var myRnd = RndGen.init(id);

        for(0 .. myWeights.len) |i| {
            myWeights[i] = (myRnd.random().float(f32)-0.5)*2/@sqrt(@as(f32, @floatFromInt(inputSize)));
        }

        const myBias = (myRnd.random().float(f32)-0.5);

        return InNeuron {
            .fanIn = inputSize,
            .layerID = myLayerID,
            .malloc = myMalloc,
            .weights = myWeights,
            .deltaWeights = myDeltas,
            .weightVelocities = myVel,
            .rmsProps = myRMSProps,
            .dwrtI = myDwrtI,
            .dwrtW = myDwrtW,
            .gradient = 0.0,
            .trainingInstances = 0.0,
            .bias = myBias,
            .deltaBias = 0.0,
            .biasRMS = 0.0,
            .biasVelocity = 0.0,
            .rnd = myRnd,
            .dropMag = 1.0,
            .age = 0,
        };

    }
    pub fn deinit(self: *StdNeuron) void {
        self.malloc.free(self.weights);
        self.malloc.free(self.deltaWeights);
        self.malloc.free(self.dwrtI);
        self.malloc.free(self.dwrtW);
        self.malloc.free(self.weightVelocities);
        self.malloc.free(self.rmsProps);
    }
};

const StdNeuron =  struct {
    comptime fanIn: usize = layerSizes,
    malloc: Allocator,
    weights: []f32,
    deltaWeights: []f32,
    weightVelocities: []f32,
    rmsProps: []f32,
    bias: f32,
    deltaBias: f32,
    biasVelocity: f32,
    biasRMS: f32,
    dwrtI: []f32,
    dwrtW: []f32,
    layerID: usize,
    gradient: f32,
    trainingInstances: f32,
    age: usize,
    rnd: std.Random.Xoshiro256,
    dropMag: f32,

    pub fn consume(self: *StdNeuron, input: []f32, dropout: bool) f32 {
//         @setFloatMode(std.builtin.FloatMode.optimized);
        self.dropMag = 1.0;
        if(dropout and self.layerID < numLayers) {
            self.dropMag = 1.0/(1.0-dropoutChance);
            if(@abs(self.rnd.random().float(f32)) < dropoutChance) {
                self.gradient = 0.0;
                return 0.0;
            }
        }

        assert(self.weights.len == input.len);

        @memcpy(self.dwrtI, self.weights);
        @memcpy(self.dwrtW, input);

        const inVec: @Vector(self.fanIn, f32) = input[0..self.fanIn].*;
        const weightVec :@Vector(self.fanIn, f32) = self.weights[0..self.fanIn].*;

        const summer = inVec * weightVec;

        const sum = @reduce(.Add, summer) * self.dropMag + self.bias;

        if(self.layerID < numLayers) {
            //TANH INNER ACTIVATIONS
//             const res = std.math.tanh(sum);
//             self.gradient = 1.0-res*res;
//             return res;
            //SNEAKY RELU INNER ACTIVATIONS
//              const res = @max(0, sum);
//              self.gradient = res / sum;
//              return res;
            //SNEAKY LEAKY RELU INNER ACTIVATIONS
//              const res = @max(0.01*sum, sum);
//              self.gradient = res / sum;
//              return res;
            //ELU INNER ACTIVATIONS
                 if(sum >= 0) {
                     self.gradient = 1.0;
                     return sum;
                 }
                 const res = @exp(sum) - 1.0;
                 self.gradient = res + 1.0;
                 return res;
            //SWISH INNER ACTIVATIONS
//              const exp = @exp(sum);
//              const sig = exp/(1.0+exp);
//              const res = sum * sig;
//              self.gradient = sig * (1.0+sum*(1.0-sig));
//              return res;
        }
        //SIGMOIDAL OUTER ACTIVATIONS
        var res = @exp(sum);
        res /= res + 1.0;
        self.gradient = res*(1.0-res);
        return res;
        //LINEAR OUTER ACTIVATIONS
        //self.gradient = 1.0;
        //return sum;
    }
    pub fn emit(self: *StdNeuron, err: f32, accum: []f32) void {
        if(self.gradient == 0.0) {
            self.trainingInstances += 1.0;
            return;
        }
//         @setFloatMode(std.builtin.FloatMode.optimized);
        var dist = err * self.gradient;

        self.deltaBias += dist;
        self.trainingInstances += 1.0;

        dist *= self.dropMag;

        const distSpray: @Vector(self.fanIn, f32) = @splat(dist);

        const inpVec: @Vector(self.fanIn, f32) = self.dwrtI[0..self.fanIn].*;
        const weiVec: @Vector(self.fanIn, f32) = self.dwrtW[0..self.fanIn].*;

        const accuVec: @Vector(self.fanIn, f32) = accum[0..self.fanIn].*;
        const deltaVec: @Vector(self.fanIn, f32) = self.deltaWeights[0..self.fanIn].*;

        const accuArray :[self.fanIn]f32 = @mulAdd(@Vector(self.fanIn, f32), distSpray, inpVec, accuVec);
        const deltaArray :[self.fanIn]f32 = @mulAdd(@Vector(self.fanIn, f32), distSpray, weiVec, deltaVec);

        @memcpy(accum, &accuArray);
        @memcpy(self.deltaWeights, &deltaArray);
    }
    pub fn update(self: *StdNeuron, rateMod: f32) void {
//         @setFloatMode(std.builtin.FloatMode.optimized);
        self.age += 1;
        const velScalar = 1.0 - std.math.pow(f32, MOMENTUM, @as(f32, @floatFromInt(self.age)));
        const rmsScalar = 1.0 - std.math.pow(f32, RMS_DECAY, @as(f32, @floatFromInt(self.age)));
        inline for(0 .. self.fanIn) |i| {
            const delta = self.deltaWeights[i] / self.trainingInstances;
            self.weightVelocities[i] = MOMENTUM * self.weightVelocities[i] + (1.0-MOMENTUM) * delta;
            self.rmsProps[i] = RMS_DECAY * self.rmsProps[i] + (1.0-RMS_DECAY) * delta * delta;
            self.weights[i] = self.weights[i] * WEIGHT_DECAY - ETA * rateMod *
                clipGradient((self.weightVelocities[i] / velScalar) / ((@sqrt(self.rmsProps[i]/rmsScalar) + 0.0001)));
        }
        @memset(self.deltaWeights, 0.0);
        const delta = self.deltaBias / self.trainingInstances;
        self.biasVelocity = MOMENTUM * self.biasVelocity + (1.0-MOMENTUM) * delta;
        self.biasRMS = RMS_DECAY * self.biasRMS + (1.0-RMS_DECAY) * delta * delta;
        self.bias = self.bias * WEIGHT_DECAY - ETA * rateMod *
            clipGradient((self.biasVelocity / velScalar) / ((@sqrt(self.biasRMS/rmsScalar) + 0.0001)));
        self.deltaBias = 0.0;
        self.trainingInstances = 0.0;
    }
    pub fn init(comptime inputSize: usize, myLayerID: usize, id: usize, myMalloc: Allocator) !StdNeuron {
        const myWeights :[]f32 = try myMalloc.alloc(f32, inputSize);
        errdefer myMalloc.free(myWeights);
        const myDeltas :[]f32 = try myMalloc.alloc(f32, inputSize);
        errdefer myMalloc.free(myDeltas);
        const myDwrtI :[]f32 = try myMalloc.alloc(f32, inputSize);
        errdefer myMalloc.free(myDwrtI);
        const myDwrtW :[]f32 = try myMalloc.alloc(f32, inputSize);
        errdefer myMalloc.free(myDwrtW);
        const myVel :[]f32 = try myMalloc.alloc(f32, inputSize);
        errdefer myMalloc.free(myVel);
        const myRMSProps :[]f32 = try myMalloc.alloc(f32, inputSize);
        errdefer myMalloc.free(myRMSProps);
        const myBVel :f32 = 0.0;
        const myGradient :f32 = 0.0;

        var myRnd = RndGen.init(id);

        for(0 .. myWeights.len) |i| {
            myWeights[i] = (myRnd.random().float(f32)-0.5)*2/@sqrt(@as(f32, @floatFromInt(inputSize)));
        }

        const myBias = (myRnd.random().float(f32)-0.5);

        return StdNeuron {
            .fanIn = inputSize,
            .layerID = myLayerID,
            .malloc = myMalloc,
            .weights = myWeights,
            .deltaWeights = myDeltas,
            .weightVelocities = myVel,
            .rmsProps = myRMSProps,
            .dwrtI = myDwrtI,
            .dwrtW = myDwrtW,
            .gradient = myGradient,
            .trainingInstances = 0.0,
            .age = 0,
            .bias = myBias,
            .deltaBias = 0.0,
            .biasRMS = 0.0,
            .biasVelocity = myBVel,
            .rnd = myRnd,
            .dropMag = 1.0,
        };
    }
    pub fn deinit(self: *StdNeuron) void {
        self.malloc.free(self.weights);
        self.malloc.free(self.deltaWeights);
        self.malloc.free(self.dwrtI);
        self.malloc.free(self.dwrtW);
        self.malloc.free(self.weightVelocities);
        self.malloc.free(self.rmsProps);
    }
};
