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
import std.math;
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
      --input       - test input to feed trained network. Default
                      is 'test.png'
";

void main(string[] args)
{
    bool learning = true;
    bool recognition = false;
    bool help = false;
    bool genconfig = false;
    string configPath = "config.json";
    string testInput = "test.png";
    
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
            "genconfig", &genconfig,
            "input", &testInput
        );
        
        if(recognition) learning = false;
        
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
    enum OUTPUT_SIZE = 4;
    alias TestNet = Perceptron!(INPUT_SIZE, INPUT_SIZE*INPUT_SIZE, 2*INPUT_SIZE, OUTPUT_SIZE);
    
    if(learning)
    {
        logger.logInfo("Application operates in learning mode");
        auto inputSet = InputSet(logger, config.learnFolder, config.learnSamples, config.controlPart.to!float, config.saveInput);
        
        logger.logInfo(text("Readed samples: ", inputSet.samples[].walkLength));
        size_t i;
        foreach(ref sample; inputSet.samples)
        {
            logger.logInfo(text("Sample ", i++));
            logger.logInfo(text("learning set length: ", sample.learnSet[].walkLength));
            logger.logInfo(text("control set length: ", sample.checkSet[].walkLength));
            logger.logInfo(text("symbol: ", sample.answer));
            logger.logInfo(text("answer vector: ", sample.answerVector));
        } 
        
        TestNet testNet;
        testNet.randomInit;
        //testNet.load(config.networkFile);
        
        double oldAcc = testNet.finalAccuracy(inputSet);
        logger.logInfo(text("Final accuracy before learning: ", oldAcc));
        testNet.learn(inputSet, config.trainingFactor.to!double, config.inertiaFactor.to!double, config.iteratesCount.to!size_t);
        
        double finalAcc = testNet.finalAccuracy(inputSet);
        logger.logInfo(text("Final accuracy after learning: ", finalAcc, " (was ", oldAcc, ")"));
        
        logger.logInfo(text("Saving trained network to ", config.networkFile));
        testNet.save(config.networkFile);
        
        // testing saving
        auto testNet2 = TestNet.load(config.networkFile);
        assert(testNet2.finalAccuracy(inputSet).approxEqual(finalAcc));
    }
    else
    {
        logger.logInfo("Application operates in recognition mode");
        
        logger.logInfo(text("Loading trained network from ", config.networkFile));
        auto testNet = TestNet.load(config.networkFile);
        
        logger.logInfo(text("Loading test sample from ", testInput));
        auto inputs = parseInput(testInput);
        debugSaveInput(testInput, "debug.png");
        
        logger.logInfo("Loading symbol map from config");
        auto symbolMap = config.symbolMap!OUTPUT_SIZE;
        
        logger.logInfo("Getting answer");
        auto answer = testNet.detectSymbol(inputs, symbolMap);
        
        logger.logInfo(text("Answer is: ", answer.symbol));
        logger.logInfo(text("Assurance is: ", answer.assurance));
    } 
}