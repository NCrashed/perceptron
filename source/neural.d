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

struct Neuron(size_t inputLength)
{
    float[inputLength] weights;
    
    enum length = inputLength;
    
    void randomInit()
    {
        foreach(ref val; weights)
        {
            val = uniform!"[]"(-5.0, 5.0);
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
        return cast(float)(val / (abs(val) + 1));
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
}

struct Layer(size_t inputLength, size_t neuronCount)
{
    Neuron!inputLength[neuronCount] neurons;
    
    enum output = neuronCount;
    enum input  = inputLength;
    
    void randomInit()
    {
        foreach(ref neuron; neurons)
        {
            neuron.randomInit();
        }
    }
    
    float[output] calculate()(auto ref float[input] inputs)
    {
        float[output] buff;
        
        foreach(i, ref val; buff)
        {
            val = neurons[i].calculate(inputs);
        }
        
        return buff;
    }
    
    Json toJson() const
    {
        Json[string] ret;
        ret["neurons"] = serializeToJson(neurons);
        return Json(ret);
    }
    
    static typeof(this) fromJson(Json src)
    {
        typeof(this) ret;
        
        ret.neurons = deserializeJson!(Neuron!inputLength[])(src.neurons);
        
        return ret;
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
    
    float[output] calculate(ubyte[input] rawInputs)
    {
        float[input] inputs;
        foreach(i, ref input; inputs)
        {
            input = cast(float)rawInputs[i];
        }
        
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