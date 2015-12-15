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

# Running in the Scala REPL

If you want to start the Scala REPL to try out some code quickly, you
can launch it using:
`./gradlew ml-app:scalaConsole -q`

The REPL includes all of the idiml code dependencies in the class
path, so this is a good way to quickly experiment with new libraries.

# Unit testing

Run unit tests with the `test` task: `./gradlew test`. Unit testing is
performed with [ScalaTest](http://www.scalatest.org/). HTML test results
for each project are stored in the project's `build/reports` subdirectory.

# Setting up Intellij

There are two ways to setup intellij:

1. `./gradlew idea` which will create intellij project files.
2. Then in intellij import an existing project from the produced project files.

or

1. Import an existing gradle project, and point it to the `build.gradle` file.

You will probably then need to tweak things to get say unit tests to work. To do that, click the top
right hand icon next to the magnifying glass. It will show you the project structure. Make sure you
have the following set:

1. Under `SDKs` that Java 1.8 is set.
2. Under `global libraries` you have the Scala SDK - 2.11.7.
3. Under `Modules` that you have for `predict`, `train` and `ml-app` the compiler output is set
   to `use module compile output path`.
4. You should now be able to run unit tests and the ml-app even.
