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

import std.math;
import std.random;
import std.conv;

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