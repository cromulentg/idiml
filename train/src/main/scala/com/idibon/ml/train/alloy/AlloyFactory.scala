package com.idibon.ml.train.alloy

import com.idibon.ml.common.Engine
import com.typesafe.scalalogging.StrictLogging
import org.json4s.native.Serialization.{read}

/**
  * Static factory class to create the right Alloy Trainer.
  */
object AlloyFactory extends StrictLogging {

  /**
    * Given the JSON String, create the correct Alloy Trainer with the right parameters.
    *
    * @param engine
    * @param jsonString
    * @return
    */
  def getTrainer(engine: Engine, jsonString: String): AlloyTrainer = {
    implicit val formats = BuilderDefaults.formats
    val builderObject = read(jsonString).asInstanceOf[AlloyTrainerBuilder]
    logger.info(s"Creating Trainer:\n $builderObject")
    builderObject.build(engine)
  }

}
