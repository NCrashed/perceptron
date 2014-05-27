/**
 * Confguration file description
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
module config;

import std.conv;
import std.stdio;
import std.range;
import std.path;
import std.file;
import vibe.data.json;

struct Config
{
    string logFile = "perceptron.log";
    string learnFolder = "data/learn";
    string recogintionFolder = "data/recognition";
    bool   saveInput = false;
    string controlPart = "0.2";
    string trainingFactor = "0.8";
    string inertiaFactor = "0.0";
    string iteratesCount = "50";
    string networkFile = "network.json";
    
    struct Sample
    {
        string path; // can be folder or file
        string symbol;
    }
    
    Sample[] learnSamples;
    Sample[] recognitionSamples;
    
    dchar[length] symbolMap(size_t length)()
    {
    	assert(learnSamples.length == length, text("Need symbol map length doesn't correspond with samples count! ", length, " != ", learnSamples.length ));
    	dchar[length] buff;
    	foreach(i, ref sample; learnSamples)
    	{
    		assert(sample.symbol.length != 0, "Empty string as symbol!");
    		buff[i] = sample.symbol[0];
    	}
    	
    	return buff;
    }
}

Config loadConfig(string path)
{
    return deserializeJson!Config(File(path, "r").byLine.join);
}

void saveConifg()(auto ref Config conf, string path)
{
    if(!path.dirName.exists)
    {
        mkdirRecurse(path.dirName);
    }
    
    auto range = new File(path, "w").lockingTextWriter;
    writePrettyJsonString(range, conf.serializeToJson, 0);
}