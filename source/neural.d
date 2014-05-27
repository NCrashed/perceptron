/**
 *  Perceptron representation.
 *
 * License:
 *   This Source Code Form is subject to the terms of
 *   the Mozilla Public License, v. 2.0. If a copy of
 *   the MPL was not distributed with this file, You
 *   can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * Authors:
 *   Anton Gushcha <ncrashed@gmail.com>
 */
module neural;

import vibe.data.json;
import std.math;
import std.random;
import std.conv;
import std.path;
import std.file;
import std.range;
import std.stdio;
import std.container;
import input;

struct Neuron(size_t inputLength)
{
    float[inputLength] weights;
    
    enum length = inputLength;
    
    void randomInit()
    {
        foreach(ref val; weights)
        {
            val = uniform!"[]"(-0.1, 0.1);
        }
    }
    
    float calculate()(auto ref float[inputLength] inputs)
    {
        float summ = 0.0;
        foreach(i, input; inputs)
        {
            summ += input * weights[i];
        }
        return activationFunc(summ);
    }
    
    private float activationFunc(float val)
    {
        return cast(float)(1.0 / (exp(-val) + 1.0));
    }
    
    Json toJson() const
    {
        Json[string] ret;
        ret["weights"] = serializeToJson(weights);
        return Json(ret);
    }
    
    static typeof(this) fromJson(Json src)
    {
        typeof(this) ret;
        ret.weights = deserializeJson!(float[])(src.weights);
        return ret;
    }
    
    private @ignore
    {
        float[inputLength] dw;
        
        void cleanupDeltas()
        {
            dw[] = 0.0;
        }
    }
}

struct Layer(size_t inputLength, size_t neuronCount)
{
    // neuronCount
    Array!(Neuron!inputLength) neurons;
    
    enum output = neuronCount;
    enum input  = inputLength;
    
    void randomInit()
    {
        foreach(i; 0 .. neuronCount)
        {
            auto neuron = Neuron!inputLength();
            neuron.randomInit();
            neurons.insert(neuron);
        }
    }
    
    float[output] calculate()(auto ref float[input] inputs)
    {
        float[output] buff;
        
        size_t i;
        foreach(ref neuron; neurons[])
        {
            buff[i++] = neuron.calculate(inputs);
        }
        
        return buff;
    }
    
    Json toJson() const
    {
        Json[string] ret;
        ret["neurons"] = serializeToJson((cast()neurons)[].array);
        return Json(ret);
    }
    
    static typeof(this) fromJson(Json src)
    {
        typeof(this) ret;
        
        auto jsonArr = src.neurons.get!(Json[]);
        foreach(json; jsonArr)
        {
            ret.neurons.insert(deserializeJson!(Neuron!inputLength)(json));
        }
                
        return ret;
    }
    
    // learning staff
    private @ignore
    {
        float[output] outputs;
        float[output] deltas;
        
        void clenupLearning()
        {
            Array!(Neuron!inputLength) newNeurons;
            foreach(neuron; neurons[])
            {
                neuron.cleanupDeltas();
                newNeurons.insert(neuron);
            }
            neurons = newNeurons;
            
            outputs[] = 0.0;
            deltas[] = 0.0;
        }
        
        void calcOutputs()(auto ref float[input] inputs)
        {
            outputs = calculate(inputs);
        }
        
        void calcDeltasEnd(float[output] t)
        {
            foreach(k, ref dk; deltas)
            {
                // dk = ok*(1 - ok)*(tk - ok)
                dk = outputs[k]*(1 - outputs[k])*(t[k] - outputs[k]);
            }
        }
        
        void calcDeltas(NextLayer)(ref NextLayer nextLayer)
        {
            static assert(output == nextLayer.input);
            
            foreach(j, ref dj; deltas)
            {
                // dj = oj*(1 - oj)* summ!(k in children(j))(dk * w[j][k])
                double summ = 0.0;
                foreach(k; 0 .. nextLayer.output)
                {
                    // dk * w[j][k]
                    summ += nextLayer.deltas[k] * nextLayer.neurons[k].weights[j];
                }
                dj = outputs[j]*(1 - outputs[j]) * summ;
            }
        }
        
        void applyDeltas()(auto ref float[input] inputs, double learningSpeed, double inertiaFactor)
        {
            Array!(Neuron!inputLength) newNeurons;
            size_t j;
            foreach(ref neuron; neurons[])
            {
                foreach(i; 0 .. input)
                {
                    neuron.dw[i] = inertiaFactor * neuron.dw[i] + (1 - inertiaFactor)*learningSpeed * deltas[j] * inputs[i];
                    neuron.weights[i] = neuron.weights[i] + neuron.dw[i];
                }
                
                newNeurons.insert(neuron);
                j++;
            }
            neurons = newNeurons;
        }
    }
}

struct Perceptron(size_t inputLength, TS...)
{
    static assert(TS.length > 0, "Neural net with zero layers!");
    mixin(genLayers!(inputLength, TS));
    
    enum input = inputLength;
    enum output = last!TS;
    enum layers = TS.length;
    
    void randomInit()
    {
        foreach(i, count; TS)
        {
            mixin(text(layer(i), ".randomInit;"));
        }
    }
    
    float[input] transformInput()(auto ref ubyte[input] rawInputs)
    {
        float[input] inputs;
        foreach(i, ref input; inputs)
        {
            input = cast(float)rawInputs[i];
        }
        return inputs;
    }
    
    float[output] calculate()(auto ref ubyte[input] rawInputs)
    {
        auto inputs = transformInput(rawInputs);
        
        string genBody()
        {
            string ret;
            foreach(i; 0 .. layers)
            {
                if(i == 0)
                {
                    ret ~= text("auto output", i, " = ", layer(i), ".calculate(inputs);\n");
                }
                else
                {
                    ret ~= text("auto output", i, " = ", layer(i), ".calculate(output", i-1, ");\n");
                }
            }
            ret ~= text("return output", layers-1, ";\n");
            return ret;
        }
        
        mixin(genBody());
    }
    
    double error()(auto ref ubyte[input] rawInputs, auto ref float[output] answerVector)
    {
        auto outputs = calculate(rawInputs);
        
        double summ = 0.0;
        foreach(i; 0 .. output)
        {
            double diff = outputs[i] - answerVector[i];
            summ += diff*diff;
        }
        
        return summ * 0.5;
    }
    
    struct Answer
    {
    	dchar symbol;
    	float assurance = 0.0f;
    }
    
    Answer detectSymbol()(auto ref ubyte[input] rawInputs, auto ref dchar[output] symbolsMap)
    {
    	auto answer = calculate(rawInputs);
    	//std.stdio.writeln(answer);
    	
    	size_t max;
    	float maxVal = -float.max;
    	foreach(i, val; answer)
    	{
    		if(val > maxVal)
    		{
    			max = i;
    			maxVal = val;
    		}
    	}
    	
    	return Answer(symbolsMap[max], maxVal);
    }
    
    double finalAccuracy()(auto ref InputSet inputSet)
    {
    	auto symbolMap = inputSet.symbolsMap;
    	size_t totalChecks, correctChecks;
    	foreach(ref sample; inputSet.samples[])
    	{
    		foreach(ref inputs; sample.learnSet[])
    		{
    			auto ans = detectSymbol(inputs, symbolMap); 
    			//std.stdio.write(ans, " ?= ");
    			//std.stdio.writeln(sample.answer);
    			if(ans.symbol == sample.answer)
    			{
    				correctChecks++;
    			} 
    			totalChecks++;
    		}
    	}
    	
    	return correctChecks / cast(double)totalChecks; 
    }
    
    Json toJson() const
    {
        string genBody()
        {
            string ret;
            foreach(i; 0 .. layers)
            {
                ret ~= text("builder.put(", layer(i), ".toJson);\n");
            }
            return ret;
        }
        
        auto builder = appender!(Json[]);
        mixin(genBody());
        
        Json[string] ret;
        ret["layers"] = builder.data;
        
        return Json(ret);
    }
    
    static typeof(this) fromJson(Json src)
    {
        typeof(this) ret;
        
        Json[] jsonLayers = src.layers.get!(Json[]);
        
        string genBody()
        {
            string ret;
            foreach(i; 0 .. layers)
            {
                ret ~= text("ret.", layer(i), " = deserializeJson!(typeof(ret.", layer(i), "))(jsonLayers[", i, "]);\n");
            }
            return ret;
        }
        
        mixin(genBody());
        
        return ret;
    }
    
    void save(string filename)
    {
        if(!filename.dirName.exists)
        {
            mkdirRecurse(filename.dirName);
        }
    
        auto file = new File(filename, "w");
        scope(exit) file.close();
        
        auto range = file.lockingTextWriter;
        writePrettyJsonString(range, serializeToJson(this), 0);  
    }
    
    static typeof(this) load(string filename)
    {
        return deserializeJson!(typeof(this))(File(filename, "r").byLine.join.idup);
    }
    
    void learn(InputSet inputSet, double learnSpeed, double inertiaFactor, size_t stepsCount)
    {
        // Clearing saved final deltas
        foreach(j, _t; TS)
        {
            mixin(layer(j)).clenupLearning();   
        }
        
        foreach(d; 0 .. stepsCount)
        {
        	auto samples = inputSet.samples[].array;
        	assert(samples.length == 4);
        	
        	auto subrange0 = zip(samples[0].learnSet[], repeat(samples[0]));
        	auto subrange1 = zip(samples[1].learnSet[], repeat(samples[1]));
        	auto subrange2 = zip(samples[2].learnSet[], repeat(samples[2]));
        	auto subrange3 = zip(samples[3].learnSet[], repeat(samples[3]));
        	auto range = roundRobin(subrange0, subrange1, subrange2, subrange3);
        	
            //foreach(ref sample; inputSet.samples[])
            {
                //foreach(ref rawInputs; sample.learnSet[])
                foreach(elem; range)
                { 
                	auto sample = elem[1];
                	auto rawInputs = elem[0];
                	//
                    double wasError = error(rawInputs, sample.answerVector[0 .. output]);
                    auto inputs = transformInput(rawInputs);
                    
                    // Calculating outputs in all neurons
                    foreach(j, _t; TS)
                    {
                        static if(j == 0)
                        {
                            mixin(layer(j)).calcOutputs(inputs);
                        }
                        else
                        {
                            mixin(layer(j)).calcOutputs(mixin(layer(j-1)).outputs);
                        }
                    }
                    
                    // Calculate deltas at output layer
                    assert(sample.answerVector.length == mixin(layer(layers-1)).output, "Answer vector and output of neural network doesn't match!");
                    mixin(layer(layers-1)).calcDeltasEnd(sample.answerVector[0 .. mixin(layer(layers-1)).output]);
                    
                    // Calculate deltas at other layers
                    foreach(_l, _t; TS[0 .. $-1]) // _l in [0 .. layers)
                    {
                        enum l = layers - 2 - _l; // l in [layers - 1 .. 0]
                        //pragma(msg, l);
                        mixin(layer(l)).calcDeltas(mixin(layer(l+1)));
                    }
                    
                    // Apply deltas for first layer
                    mixin(layer(0)).applyDeltas(inputs, learnSpeed, inertiaFactor);
                    
                    // Apply deltas for other layers
                    foreach(_l, _t; TS[1 .. $])
                    {
                        enum l = _l + 1;
                        //pragma(msg, l);
                        mixin(layer(l)).applyDeltas(mixin(layer(l-1)).outputs, learnSpeed, inertiaFactor);
                    }
                    
                    double nowError = error(rawInputs, sample.answerVector[0 .. output]);
                    std.stdio.writeln("Error: ", wasError, " -> ", nowError);
                }
            }
        }
    }
}

private string layer(size_t i)
{
    return text("layer", i);
}

private string genLayer(size_t i, size_t inputLength, size_t neuronCount)
{
    return text("Layer!(", inputLength, ",", neuronCount,") layer", i, ";");
}

private template genLayers(size_t inputLength, TS...)
{
    private template innerGenLayers(size_t i, size_t inputLength, size_t neuronCount, TS...)
    {
        static if(TS.length == 0)
        {
            enum innerGenLayers = genLayer(i, inputLength, neuronCount) ~ "\n";
        }
        else
        {
            enum innerGenLayers = genLayer(i, inputLength, neuronCount) ~ "\n"
                ~ innerGenLayers!(i+1, neuronCount, TS[0], TS[1 .. $]);
        }
    }
    
    enum genLayers = innerGenLayers!(0, inputLength, TS[0], TS[1 .. $]);
}

private template last(TS...)
{
    static if(TS.length == 1)
    {
        enum last = TS[0];
    }
    else
    {
        enum last = last!(TS[1 .. $]);
    }
}