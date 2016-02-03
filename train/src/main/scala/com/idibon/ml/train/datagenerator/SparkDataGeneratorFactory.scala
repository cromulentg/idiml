package com.idibon.ml.train.datagenerator

import com.idibon.ml.train.datagenerator.BuilderDefaults
import com.typesafe.scalalogging.StrictLogging
import org.json4s.native.Serialization.{read}

/**
  * Static factory class to create the right furnace.
  */
object SparkDataGeneratorFactory extends StrictLogging {

  /**
    * Given the JSON String, create the correct data generator with the right parameters.
    *
    * @param jsonString
    * @return
    */
  def getDataGenerator(jsonString: String): SparkDataGenerator = {
    implicit val formats = BuilderDefaults.formats
    val builderObject = read(jsonString).asInstanceOf[SparkDataGeneratorBuilder]
    logger.info(s"Creating SparkDataGenerator:\n $builderObject")
    builderObject.build()
  }

}
