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

enum helpMsg = 
"perceptron [args]

args: --learning    - defines learning mode for neural network
      --recognition - defines recognition mode for neural networ.
                      default flag.
      --config=path - configuration file, default is 'config.json'.
";

void main(string[] args)
{
    bool learning = false;
    bool recognition = true;
    bool help = false;
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
            "help", &help
        );
        
        assert(learning != recognition, "Cannot use learning and recognition at the same time!");
        
        if(help)
        {
            writeln(helpMsg);
            return;
        }
    }
    
    
}