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
    
    struct Sample
    {
        string path; // can be folder or file
        string symbol;
    }
    
    Sample[] learnSamples;
    Sample[] recognitionSamples;
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