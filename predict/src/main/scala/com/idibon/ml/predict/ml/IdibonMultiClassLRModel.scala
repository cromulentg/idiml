package com.idibon.ml.predict.ml

import java.io.{IOException, DataInputStream, DataOutputStream}

import com.idibon.ml.alloy.Alloy.{Writer, Reader}
import com.idibon.ml.alloy.Codec
import com.idibon.ml.common.{Archivable, ArchiveLoader, Engine}
import com.idibon.ml.feature.{Feature, FeaturePipeline}
import com.idibon.ml.predict._
import com.typesafe.scalalogging.StrictLogging
import org.apache.spark.mllib.classification.IdibonSparkMLLIBLRWrapper
import org.apache.spark.mllib.linalg.{Vectors, Vector}
import org.json4s._


/**
  * Class that wraps our extended Spark MLLIB Multinomial LR model.
  *
  * This means that this class returns a MultiLabelDocumentResult and
  * it should be used with a GangModel, rather than an EnsembleModel.
  *
  * @author "Stefan Krawczyk <stefan@idibon.com>"
  */
case class IdibonMultiClassLRModel(labelToInt: Map[String, Int],
  lrm: IdibonSparkMLLIBLRWrapper,
  override val featurePipeline: Option[FeaturePipeline])
    extends MLModel[Classification](featurePipeline) with StrictLogging
    with Archivable[IdibonMultiClassLRModel, IdibonMultiClassLRModelLoader] {

  val intToLabel = labelToInt.map({case (label, index) => (index, label)})

  val reifiedType = classOf[IdibonMultiClassLRModel]

  /**
    * The model will use a subset of features passed in. This method
    * should return the ones used.
    *
    * @return Vector (likely SparseVector) where indices correspond to features
    *         that were used.
    */
  def getFeaturesUsed(): Vector = {
    return lrm.getFeaturesUsed()
  }

  /**
    * This method returns the metric that best represents model quality after training
    *
    * @return Double (e.g. AreaUnderROC)
    */
  def getEvaluationMetric(): Double = ???

  /**
    * The method used to predict from a vector of features.
    *
    * @param featVec Vector of features to use for prediction.
    * @param invertFeatureFn
    * @param options  Object of predict options.
    * @return
    */
  override def predictVector(featVec: Vector,
    invertFeatureFn: (Vector) => Seq[Option[Feature[_]]],
    options: PredictOptions): Seq[Classification] = {

    val results = lrm.predictProbability(featVec).toArray

    // map of label index to significant features for that label
    val significantFeatures = if (options.includeSignificantFeatures) {
      lrm.getSignificantDimensions(featVec, options.significantFeatureThreshold)
        .map({ case (labelIndex, sigDimensions) => {
          (labelIndex, CanHazPipeline.zipFeaturesAndWeights(sigDimensions,
            invertFeatureFn(sigDimensions)))
        }}).toMap
    } else {
      Map[Int, Seq[(Feature[_], Float)]]()
    }

    // generate a classification result for each result
    results.zipWithIndex.map({ case (probability, labelIndex) => {
      Classification(intToLabel(labelIndex), probability.toFloat,
        1, PredictResultFlag.NO_FLAGS,
        significantFeatures.get(labelIndex).getOrElse(Seq[(Feature[_], Float)]())
      )
    }}).sortWith(_.probability > _.probability)
  }

  /** Serializes the object within the Alloy
    *
    * Implementations are responsible for persisting any internal state
    * necessary to re-load the object (for example, feature-to-vector
    * index mappings) to the provided Alloy.Writer.
    *
    * Implementations may return a JObject of configuration data
    * to include when re-loading the object.
    *
    * @param writer destination within Alloy for any resources that
    *               must be preserved for this object to be reloadable
    * @return Some[JObject] of configuration data that must be preserved
    *         to reload the object. None if no configuration is needed
    */
  override def save(writer: Writer): Option[JObject] = {
    val coeffs = writer.within("model").resource("coefficients.libsvm")
    IdibonMultiClassLRModel.writeCodecLibSVM(
      this.labelToInt, coeffs, this.lrm.intercept, this.lrm.weights, this.lrm.numFeatures)
    coeffs.close()
    //TODO: store other model metadata like training date, etc.
    Some(new JObject(List(
      JField("version", JString(IdibonMultiClassLRModel.FORMAT_VERSION)),
      savePipelineIfPresent(writer)
    )))
  }

  /**
    * Returns a training summary. You have to override this to actually return something.
    *
    * @return
    */
  override def getTrainingSummary(): Option[Seq[TrainingSummary]] = {
    return trainingSummary
  }
}

/**
  * Static object that houses static functions and constants.
  */
object IdibonMultiClassLRModel extends StrictLogging {
  val FORMAT_VERSION = "0.0.3"

  /**
    * Static method to write our "libsvm" like format to a stream.
    *
    * @param labelToInt
    * @param out
    * @param intercept
    * @param coefficients
    */
  def writeCodecLibSVM(labelToInt: Map[String, Int],
                       out: DataOutputStream,
                       intercept: Double,
                       coefficients: Vector,
                       numFeatures: Int): Unit = {
    logger.info(s"Writing ${coefficients.size} dimensions with " +
      s"${coefficients.numNonzeros} active dimensions with $intercept for MultiClass LR")
    // int to label map
    // size
    Codec.VLuint.write(out, labelToInt.size)
    labelToInt.foreach({case (label, index) => {
      // label
      Codec.String.write(out, label)
      // int index
      Codec.VLuint.write(out, index)
    }})
    // number of features
    Codec.VLuint.write(out, numFeatures)
    // intercept
    out.writeDouble(intercept)
    // dimensions
    Codec.VLuint.write(out, coefficients.size)
    // actual non-zero dimensions
    Codec.VLuint.write(out, coefficients.numNonzeros)
    var maxCoefficient = -10000.0
    var minCoefficient = 10000.0
    var lastIndex = 0
    coefficients.foreachActive({ case (index, value) => {
        if (value != 0.0) {
          // skip over zero values
          Codec.VLuint.write(out, index - lastIndex)
          lastIndex = index
          out.writeDouble(value)
          if (value > maxCoefficient) maxCoefficient = value
          if (value < minCoefficient) minCoefficient = value
        }
    }})
  }

  /**
    * Static method to read our "libsvm" like format from a stream.
    *
    * @param in
    * @return
    */
  def readCodecLibSVM(in: DataInputStream): (Double, Vector, Map[String, Int], Int) = {
    // int to label map
    val numClasses = Codec.VLuint.read(in)
    val labelToInt = (0 until numClasses).map(_ => {
      (Codec.String.read(in), Codec.VLuint.read(in))
    }).toMap
    // number of features
    val numFeatures = Codec.VLuint.read(in)
    // intercept
    val intercept = in.readDouble()
    // dimensions
    val dimensions = Codec.VLuint.read(in)
    // non-zero dimensions
    val numCoeffs = Codec.VLuint.read(in)
    // preallocate the weight vector
    val coeffs = new Array[Double](dimensions)

    var indexValue = 0
    (0 until numCoeffs).foreach({ _ => {
      val (delta, coeff) = (Codec.VLuint.read(in), in.readDouble())
      var endIndex = indexValue + delta
      (indexValue + 1 until endIndex).foreach(i => coeffs(i) = 0.0)
      coeffs(endIndex) = coeff
      indexValue = endIndex
    }})

    logger.info(s"Read $numCoeffs dimensions from $dimensions for Multiclass with intercept $intercept")
    (intercept, Vectors.dense(coeffs), labelToInt, numFeatures)
  }
}

class IdibonMultiClassLRModelLoader
  extends ArchiveLoader[IdibonMultiClassLRModel] with StrictLogging {
  /** Reloads the object from the Alloy
    *
    * @param engine implementation of the Engine trait
    * @param reader location within Alloy for loading any resources
    *               previous preserved by a call to
    *               { @link com.idibon.ml.feature.Archivable#save}
    * @param config archived configuration data returned by a previous
    *               call to { @link com.idibon.ml.feature.Archivable#save}
    * @return this object
    */
  override def load(engine: Engine, reader: Option[Reader], config: Option[JObject]): IdibonMultiClassLRModel = {
    implicit val formats = DefaultFormats
    val version = (config.get \ "version" ).extract[String]
    version match {
      case IdibonMultiClassLRModel.FORMAT_VERSION =>
        logger.info(s"Attemping to load version [v. $version] for multiclass LR.")
      case _ => throw new IOException(s"Unable to load, unhandled version [v. $version] for multiclass LR.")
    }
    val coeffs = reader.get.within("model").resource("coefficients.libsvm")
    val (intercept: Double,
         coefficients: Vector,
         labelToInt: Map[String, Int],
         numFeatures: Int) = IdibonMultiClassLRModel.readCodecLibSVM(coeffs)
    coeffs.close()
    val pipeline = CanHazPipeline.loadPipelineIfPresent(engine, reader, config)
    new IdibonMultiClassLRModel(labelToInt,
      new IdibonSparkMLLIBLRWrapper(coefficients, intercept, numFeatures, labelToInt.size),
      pipeline)
  }
}



