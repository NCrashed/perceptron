/**
 * Some utils to work with files and strings.
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
module util;

import std.stdio;

void toFile(ubyte[] bytes, string name)
{
    auto file = File(name, "w");
    scope(exit) file.close();
    
    file.rawWrite(bytes);
}
