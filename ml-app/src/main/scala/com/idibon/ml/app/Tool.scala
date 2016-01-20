package com.idibon.ml.app

import com.idibon.ml.common.Engine

/** Adds a command-line tool within the ml-app harness */
trait Tool {

  /** Executes the tool
    *
    * @param engine - the Idiml Engine context to use
    * @param argv - command-line options to configure tool
    */
  def run(engine: Engine, argv: Array[String])
}
