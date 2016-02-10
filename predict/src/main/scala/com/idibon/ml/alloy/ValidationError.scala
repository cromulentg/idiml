package com.idibon.ml.alloy

/**
  * Exception to throw with details when the model doesn't validate against itself.
  * @param msg
  */
class ValidationError(msg: String) extends Exception(msg)
