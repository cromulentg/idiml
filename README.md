Idibon ML Core
=========

Download the Gradle 2.9 binaries from http://gradle.org/gradle-download/

To install on your laptop, unzip the package and copy the `gradle-2.9` directory to `/usr/local/lib`, then create a symbolic link in `/usr/local/bin`

```
sudo cp -r ~/Downloads/gradle-2.9 /usr/local/lib
cd /usr/local/bin
sudo ln -s ../lib/gradle-2.9/bin/gradle gradle
```
