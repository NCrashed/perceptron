/**
 *  Entry point of the application. 
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
module app;

import std.stdio;
import std.getopt;
import std.conv;
import std.range;
import dlogg.strict;
import config;
import input;
import neural;

enum helpMsg = 
"perceptron [args]

args: --learning    - defines learning mode for neural network
      --recognition - defines recognition mode for neural networ.
                      default flag.
      --config=path - configuration file, default is 'config.json'.
      --genconfig   - if set, new config file is generated in 
                      --config location + .example extention.
";

void main(string[] args)
{
    bool learning = true;
    bool recognition = false;
    bool help = false;
    bool genconfig = false;
    string configPath = "config.json";
    
    {
        scope(failure)
        {
            writeln(helpMsg);
            return;
        }
        
        getopt(args,
            "learning", &learning,
            "recoginition", &recognition,
            "config", &configPath,
            "help", &help,
            "genconfig", &genconfig
        );
        
        assert(learning != recognition, "Cannot use learning and recognition at the same time!");
        
        if(help)
        {
            writeln(helpMsg);
            return;
        }
    }
    
    // If user wants to generate config
    if(genconfig)
    {
        saveConifg(Config(), configPath~".example");
        return;
    }
        
    // Parsing config
    auto config = loadConfig(configPath);
    
    // Loading logger
    shared ILogger logger = new shared StrictLogger(config.logFile);
    
    logger.logInfo("Start initialization is finished");
    
    if(learning)
    {
        logger.logInfo("Application operates in learning mode");
        auto inputSet = InputSet(logger, config.learnFolder, config.learnSamples, config.controlPart.to!float, config.saveInput);
        
        writeln("Readed samples: ", inputSet.samples[].walkLength);
        size_t i;
        foreach(ref sample; inputSet.samples)
        {
            writeln("Sample ", i++);
            writeln("learning set: ", sample.learnSet[]);
            writeln("control set: ", sample.checkSet[]);
            writeln("symbol: ", sample.answer);
        } 
        
        alias TestNet = Perceptron!(INPUT_SIZE, INPUT_SIZE*INPUT_SIZE, INPUT_SIZE*2, 3);
        TestNet testNet;
        testNet.randomInit;
        
        writeln(testNet.calculate(inputSet.samples.front.learnSet.front));
    }
    else
    {
        logger.logInfo("Application operates in recognition mode");
    } 
}