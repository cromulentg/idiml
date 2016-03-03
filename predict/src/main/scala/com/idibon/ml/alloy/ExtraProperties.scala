package com.idibon.ml.alloy

import com.idibon.ml.predict._
import com.idibon.ml.feature._
import com.idibon.ml.predict.ml.TrainingSummary

import org.json4s._
import org.json4s.native.JsonMethods

/*
 * File containing Optional traits that an Alloy could implement.
 */

/** Optional Alloy trait for Furnaces to include validation examples
  */
trait HasValidationData {
  /** List of JSON documents to use for validation */
  def validationExamples: Seq[JObject]
}

/** Optional Alloy trait for storing training configuration data */
trait HasTrainingConfig {
  /** JSON object containing the training configuration */
  def trainingConfig: JObject
}

/** Optional Alloy trait for storing training summaries */
trait HasTrainingSummary {
  def getTrainingSummaries: Option[Seq[TrainingSummary]] = None
}

/** Companion object for HasValidationData */
object HasValidationData {

  val VALIDATION_RESOURCE = "validation.dat"

  /** Loads the validation examples from the alloy and checks for correctness
    *
    * @throws ValidationError if the predicted results are invalid
    */
  def validate[T <: PredictResult with Buildable[T, Builder[T]]](
      reader: Alloy.Reader, alloy: BaseAlloy[T]) {
    val rawResource = reader.resource(VALIDATION_RESOURCE)
    if (rawResource == null) return
    val resource = new FeatureInputStream(rawResource)

    try {
      val exampleCount = Codec.VLuint.read(resource)
      val modelCount = Codec.VLuint.read(resource)
      // load model names
      val modelNames = (0 until modelCount).map(_ => Codec.String.read(resource))
      // for each validation example we have
      (0 until exampleCount).foreach(_ => {
        // get the document
        val docJson = JsonMethods.parse(Codec.String.read(resource))
        val doc = Document.document(docJson.asInstanceOf[JObject])
        // for each model
        modelNames.foreach(n => {
          // load expected results
          val expectCount = Codec.VLuint.read(resource)
          val expected = (0 until expectCount).map(_ => {
            resource.readBuildable.asInstanceOf[T]
          })
          // get fresh prediction
          val results = alloy.models(n).predict(doc, PredictOptions.DEFAULT)
          // compare number of results
          if (results.size != expected.size)
            throw new ValidationError(s"${alloy.name} ${n}")
          // compare with expected
          results.zip(expected).foreach({ case (actual, gold) => {
            if (!actual.isCloseEnough(gold))
              throw new ValidationError(s"${alloy.name} ${n}")
          }})
        })
      })
    } finally {
      resource.close()
    }
  }

  /** Saves an alloy with validation data to a persistent device
    *
    * @param writer alloy writer
    * @param alloy the alloy (with validation data) being saved
    */
  def save[T <: PredictResult with Buildable[T, Builder[T]]](
      writer: Alloy.Writer, alloy: BaseAlloy[T] with HasValidationData) {

    val resource = new FeatureOutputStream(writer.resource(VALIDATION_RESOURCE))
    val modelNames = alloy.models.keys

    try {
      Codec.VLuint.write(resource, alloy.validationExamples.size)
      Codec.VLuint.write(resource, modelNames.size)
      modelNames.foreach(n => Codec.String.write(resource, n))

      alloy.validationExamples.foreach(doc => {
        // save document
        Codec.String.write(resource, JsonMethods.compact(JsonMethods.render(doc)))
        val document = Document.document(doc)
        // get predictions
        modelNames.foreach(n => {
          val results = alloy.models(n).predict(document, PredictOptions.DEFAULT)
          Codec.VLuint.write(resource, results.size)
          results.foreach(p => resource.writeBuildable(p))
        })
      })
    } finally {
      resource.close()
    }
  }
}

/** Companion object for HasTrainingConfig */
object HasTrainingConfig {
  val CONFIG_RESOURCE = "training.json"

  /**
    * Saves the training configuration file used to the alloy.
    *
    * @param writer
    * @param alloy
    */
  def save(writer: Alloy.Writer, alloy: HasTrainingConfig) {
    val config = JsonMethods.compact(JsonMethods.render(alloy.trainingConfig))
    val resource = writer.resource(CONFIG_RESOURCE)
    try {
      Codec.String.write(resource, config)
    } finally {
      resource.close()
    }
  }
}

/** Companion object for HasTrainingSummary -- it saves and loads them */
object HasTrainingSummary {

  val TRAINING_SUMMARY_RESOURCE = "training-summary.json"

  /**
    * Gets a sequence of training summaries from the alloy if they exist.
    *
    * @param reader
    * @return sequence of training summaries.
    */
  def get(reader: Alloy.Reader): Seq[TrainingSummary] = {
    val readerResource = reader.resource(TRAINING_SUMMARY_RESOURCE)
    if (readerResource == null) return Seq()
    val resource = new FeatureInputStream(readerResource)
    try {
      val size = Codec.VLuint.read(resource)
      (0 until size).map(_ => resource.readBuildable.asInstanceOf[TrainingSummary])
    } finally {
      resource.close()
    }
  }

  /**
    * Saves the training summaries if they exist to a resource in the alloy.
    *
    * @param writer
    * @param alloy
    * @tparam T
    */
  def save[T <: PredictResult with Buildable[T, Builder[T]]](
      writer: Alloy.Writer, alloy: BaseAlloy[T] with HasTrainingSummary): Unit = {
    val summaries = getSummaries(alloy)
    // only save if we have some
    if (summaries.size < 1) return
    val resource = new FeatureOutputStream(writer.resource(TRAINING_SUMMARY_RESOURCE))
    try {
      Codec.VLuint.write(resource, summaries.size)
      summaries.foreach(summary => resource.writeBuildable(summary))
    } finally {
      resource.close()
    }
  }

  /**
    * Helper method to get training summaries from an alloy.
    * @param alloy
    * @tparam T
    * @return
    */
  def getSummaries[T <: PredictResult with Buildable[T, Builder[T]]](
      alloy: BaseAlloy[T] with HasTrainingSummary): Seq[TrainingSummary] = {
    val modelNames = alloy.models.keys
    // get summaries
    val summaries = alloy.getTrainingSummaries match {
      case None => {
        modelNames.map(n => {alloy.models(n).getTrainingSummary()})
          .collect { case Some(summary) => summary }.flatten
      }
      case Some(summaries) => summaries
    }
    summaries.toSeq
  }

}
