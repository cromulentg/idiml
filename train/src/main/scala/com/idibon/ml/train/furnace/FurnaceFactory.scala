package com.idibon.ml.train.furnace

import com.idibon.ml.common.Engine
import com.idibon.ml.predict.PredictResult
import com.typesafe.scalalogging.StrictLogging
import org.json4s.native.Serialization.{read}

/**
  * Static factory class to create the right furnace.
  */
object FurnaceFactory extends StrictLogging {

  /**
    * Given the JSON String, create the correct furnace with the right parameters.
    *
    * @param engine
    * @param jsonString
    * @return
    */
  def getFurnace[T <: PredictResult](engine: Engine, jsonString: String): Furnace[T] = {
    implicit val formats = BuilderDefaults.formats
    val builderObject = read(jsonString).asInstanceOf[FurnaceBuilder[T]]
    logger.info(s"Creating furnace:\n $builderObject")
    builderObject.build(engine)
  }

}
