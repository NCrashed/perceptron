/**
 * Converts images to byte vectors for input for neural network.
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
module input;

import ae.utils.graphics.view;
import ae.utils.graphics.sdlimage;
import ae.utils.graphics.image;
import ae.utils.graphics.color;
import dlogg.log;
import std.algorithm;
import std.container;
import std.file;
import std.range;
import std.path;
import util;
import config;

enum INPUT_SIZE_X = 5;
enum INPUT_SIZE_Y = 7;
enum INPUT_SIZE = INPUT_SIZE_X * INPUT_SIZE_Y;

struct InputSet
{   
    static struct Sample
    {
        DList!(ubyte[INPUT_SIZE]) learnSet;
        DList!(ubyte[INPUT_SIZE]) checkSet;
        dchar answer;
        float[] answerVector;
    }
    
    DList!Sample samples;
    
    this(shared ILogger logger, string baseFolder, const Config.Sample[] inputs, float controlPart, bool saveDebugInput = false)
    {
        ubyte[INPUT_SIZE] readPng(string filename)
        {
            if(filename.extension == ".png")
            {
                if(saveDebugInput)
                {
                    string newName = buildPath(filename.dirName, baseName(stripExtension(filename)) ~ "_debug.png");
                    debugSaveInput(filename, newName);
                }
                return parseInput(filename);
            }
            else
            {
                throw new Exception("Expected png file, but got "~filename.extension);
            }
        }
        
        foreach(ref input; inputs)
        {
            string inputPath = buildPath(baseFolder, input.path);
            
            if(inputPath.exists)
            {
                try
                {
                    Sample sample;
                    
                    if(inputPath.isDir)
                    {
                        auto getPngFiles() { return filter!`endsWith(a.name,".png")`(dirEntries(inputPath, SpanMode.depth)); }
                        
                        auto pngFiles = getPngFiles;
                        size_t length = pngFiles.walkLength;
                        size_t learnCount = cast(size_t)(length * controlPart);
                        
                        pngFiles = getPngFiles;
                        size_t i;
                        foreach(png; pngFiles)
                        {
                            if(i++ < learnCount)
                            {
                                sample.checkSet.insert(readPng(png.name));
                            }
                            else
                            {
                                sample.learnSet.insert(readPng(png.name));
                            }
                        }
                    }
                    else
                    {
                        sample.learnSet.insert(readPng(inputPath));
                    }
                    assert(input.symbol.length == 1, "Expected one symbol as answer!");
                    sample.answer = input.symbol[0];
                    samples.insert(sample);
                }
                catch(Exception e)
                {
                    logger.logError(text("Failed to load ", inputPath, ": ", e.msg));
                }
            }
            else
            {
                logger.logWarning(text("Cannot find folder or image by path '", inputPath, "'"));
            }
        }
        
        recalcIdealOutputs();
    } 
    
    void recalcIdealOutputs()
    {
        DList!Sample newSamples;
        size_t length = samples[].walkLength;
        size_t i;
        foreach(sample; samples[])
        {
            auto vec = new float[length];
            vec[] = 0.0;
            vec[i++] = 1.0;
            sample.answerVector = vec;
            newSamples.insert(sample); 
        }
        
        samples = newSamples;
    }
}

ubyte[INPUT_SIZE] parseInput(string fileName)
{
    auto img = transform(fileName);
    ubyte[INPUT_SIZE] buff;
    
    foreach(ix; 0 .. img.w)
    {
        foreach(iy; 0 .. img.h)
        {
            buff[ix + INPUT_SIZE_X*iy] = (img[ix, iy].l == 0) ? 0 : 1;
        }
    }
    
    return buff;
} 

void debugSaveInput(string source, string dist)
{
    transform(source).toPNG.toFile(dist);
}

private:

import std.math;

/// Present a resized view
auto downscale(V)(auto ref V src, int newW, int newH)
    if (isView!V && is(ViewColor!V == RGBX))
{
    alias COLOR = ViewColor!V;
    
    static struct Downscale
    {
        V source;
        
        int w;
        int h;

        COLOR opIndex(int x, int y)
        {
            double xfactor = source.w / w;
            double yfactor = source.h / h;
            
            int x0 = cast(int)(xfactor * x), x1 = cast(int)(xfactor * (x+1));
            int y0 = cast(int)(yfactor * y), y1 = cast(int)(yfactor * (y+1));
            
            double r = 0, g = 0, b = 0;
            foreach(ix; x0 .. x1)
            {
                foreach(iy; y0 .. y1)
                {
                   r += source[ix, iy].r;
                   g += source[ix, iy].g;
                   b += source[ix, iy].b;
                }
            }
            double size = (x1 - x0) * (y1 - y0);
            r = cast(double)round(r / size);
            g = cast(double)round(g / size);
            b = cast(double)round(b / size);
            
            return COLOR(cast(ubyte)r, cast(ubyte)g, cast(ubyte)b);
        }        
    }

    return Downscale(src, newW, newH);
}

auto grayscale(V)(auto ref V src)
        if (isView!V && is(ViewColor!V == RGBX))
{
    return src.colorMap!( c => L16(cast(ushort)(ushort.max * (.2126 * (c.r / cast(double) ubyte.max) + 0.7152 * (c.g / cast(double) ubyte.max) + 0.0722 * (c.b / cast(double) ubyte.max)))) );
}

auto otsuBinarization(V)(auto ref V src)
        if (isView!V && is(ViewColor!V == L16))
{   
    ushort threshold = otsuThreshold(src);
    return src.colorMap!( c => L16(c.l >= threshold ? ushort.max : 0) );
}

ushort otsuThreshold(V)(auto ref V src)
    if (isView!V && is(ViewColor!V == L16))
{
    ushort min = src[0, 0].l, max = src[0, 0].l;
    uint[] hist;
 
    double sigma, maxSigma=-1;
    double w1,a;
 
    /**** Построение гистограммы ****/

    // Узнаем наибольший и наименьший полутон
    foreach(ix; 0 .. src.w)
    {
        foreach(iy; 0 .. src.h)
        {
            auto temp = src[ix, iy].l;
            if(temp < min) min = temp;
            if(temp > max) max = temp;
        }
    }

    hist = new uint[max-min+1];

    // Считаем сколько каких полутонов
    foreach(ix; 0 .. src.w)
    {
        foreach(iy; 0 .. src.h)
        {
            hist[src[ix, iy].l-min]++;
        }
    }
      
 
    /**** Гистограмма построена ****/
 
    ulong temp, temp1, alpha, beta, threshold;
    /* Для расчета математического ожидания первого класса */
    foreach(i, val; hist)
    {
        temp += i*val;
        temp1 += val;
    }
 
    // Основной цикл поиска порога
    // Пробегаемся по всем полутонам для поиска такого, при котором внутриклассовая дисперсия минимальна
 
    foreach(i, val; hist)
    {
        alpha += i*val;
        beta += val;
 
        w1 = cast(double)beta / temp1;
        a = cast(double)alpha / beta - cast(double)(temp - alpha) / (temp1 - beta);
        sigma = w1*(1-w1)*a*a;
 
        if(sigma > maxSigma)
        {
            maxSigma=sigma;
            threshold=i;
        }
    }
   
    return cast(ushort)(threshold + min);
}

auto transform(string fileName)
{
    return loadImage(fileName).downscale(INPUT_SIZE_X, INPUT_SIZE_Y).grayscale.otsuBinarization;
}