package com.idibon.ml.train.alloy

import com.idibon.ml.common.Engine
import com.typesafe.scalalogging.StrictLogging
import org.json4s.JsonAST.JObject
import org.json4s.native.Serialization.{read, write}

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

  /**
    * Given a JObject, create the correct Alloy Trainer with the right parameters.
    *
    * (We just write the JObject back out to string again, since `read` doesn't work
    * on JObjects.
    *
    * @param engine
    * @param jObject
    * @return
    */
  def getTrainer(engine: Engine, jObject: JObject): AlloyTrainer = {
    implicit val formats = BuilderDefaults.formats
    getTrainer(engine, write(jObject))
  }

}
