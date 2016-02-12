package com.idibon.ml.train.datagenerator.scales

import com.typesafe.scalalogging.StrictLogging
import org.json4s.native.Serialization.read

/**
  * Static factory class to create the right furnace.
  */
object DataSetScaleFactory extends StrictLogging {

  /**
    * Given the JSON String, create the correct DataSetScale
    *
    * @param jsonString
    * @return
    */
  def getDataSetScale(jsonString: String): DataSetScale = {
    implicit val formats = BuilderDefaults.formats
    val builderObject = read(jsonString).asInstanceOf[DataSetScaleBuilder]
    logger.info(s"Creating data-set scale:\n $builderObject")
    builderObject.build()
  }

}
