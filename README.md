Idibon ML Core
=========

To get started, download and install JDK 8 from Oracle:
http://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html

Then, install Gradle 2.9 using the wrapper provided in this repository

```
idiml.git$ ./gradlew dependencies
```

# Gradle Introduction

To see a list of all supported build tasks, run `./gradlew tasks` from
the command line.

Tasks can be specified for individual projects by prefixing the task
with the project name, e.g. `./gradlew predict:jar` to run the `jar`
task within the `predict` project.

# Compilation

You can build all of the projects by executing the `jar` task, either
individually (like in the introduction above), or for all tasks:
`./gradlew jar`

# Unit testing

Run unit tests with the `test` task: `./gradlew test`. Unit testing is
performed with [ScalaTest](http://www.scalatest.org/). HTML test results
for each project are stored in the project's `build/reports` subdirectory.
