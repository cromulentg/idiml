package com.idibon.ml.predict.ml


import java.io.{DataInputStream, DataOutputStream, IOException}

import com.idibon.ml.alloy.Alloy.{Writer, Reader}
import com.idibon.ml.common.{Archivable, ArchiveLoader, Engine}
import com.idibon.ml.alloy.Codec
import com.idibon.ml.predict._
import com.idibon.ml.feature.{Feature, FeaturePipelineLoader, FeaturePipeline}
import com.typesafe.scalalogging.StrictLogging
import org.apache.spark.ml.classification.IdibonSparkLogisticRegressionModelWrapper
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.json4s._

/**
  * @author "Stefan Krawczyk <stefan@idibon.com>"
  *
  * This class implements our LogisticRegressionModel.
  */
case class IdibonLogisticRegressionModel(label: String,
  lrm: IdibonSparkLogisticRegressionModelWrapper,
  override val featurePipeline: Option[FeaturePipeline])
    extends MLModel[Classification](featurePipeline) with StrictLogging
    with Archivable[IdibonLogisticRegressionModel, IdibonLogisticRegressionModelLoader] {

  /** The method used to predict from a vector of features.
    *
    * @param featureVec Vector of features to use for prediction.
    * @param invertFeatureFn function to map vector dimensions to features
    * @param options Object of predict options.
    * @return a single PredictResult for the label classified by this model
    */
  override def predictVector(featureVec: Vector,
    invertFeatureFn: (Vector) => Seq[Option[Feature[_]]],
    options: PredictOptions): Seq[Classification] = {

    /* get the result of 1, the positive class we're interested in.
     * 0 will be 1.0 minus this value. */
    val probability = lrm.predictProbability(featureVec)(1).toFloat

    val significantFeatures = if (options.includeSignificantFeatures) {
      val sigDimensions = lrm.getSignificantDimensions(featureVec,
        options.significantFeatureThreshold)

      CanHazPipeline.zipFeaturesAndWeights(sigDimensions,
        invertFeatureFn(sigDimensions))
    } else {
      Seq[(Feature[_], Float)]()
    }

    // FIXME: return number of matched features in matchCount, not 1
    Seq(Classification(this.label, probability,
      1, PredictResultFlag.NO_FLAGS, significantFeatures))
  }

  /**
    * The model will use a subset of features passed in. This method
    * should return the ones used.
    *
    * @return Vector (likely SparseVector) where indices correspond to features
    *         that were used.
    */
  override def getFeaturesUsed(): Vector = {
    //For the predict case this is fine I believe, since we should not have 0 valued features.
    lrm.coefficients
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
    IdibonLogisticRegressionModel.writeCodecLibSVM(
      coeffs, this.lrm.intercept, this.lrm.coefficients, this.label)
    coeffs.close()
    //TODO: store other model metadata like training date, etc.
    Some(new JObject(List(
      JField("label", JString(this.label)),
      JField("version", JString(IdibonLogisticRegressionModel.FORMAT_VERSION)),
      savePipelineIfPresent(writer)
    )))
  }
}

object IdibonLogisticRegressionModel extends StrictLogging {

  val FORMAT_VERSION = "0.0.3"

  /**
    * Static method to write our "libsvm" like format to a stream.
    * @param out
    * @param intercept
    * @param coefficients
    * @param uid
    */
  def writeCodecLibSVM(out: DataOutputStream,
                       intercept: Double,
                       coefficients: Vector,
                       uid: String): Unit = {
    logger.info(s"Writing ${coefficients.size} dimensions with " +
      s"${coefficients.numActives} active dimensions with $intercept for $uid")
    // uid
    Codec.String.write(out, uid)
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
    * @param in
    * @return
    */
  def readCodecLibSVM(in: DataInputStream): (Double, Vector, String) = {
    // uid
    val uid = Codec.String.read(in)
    // intercept
    val intercept = in.readDouble()
    // dimensions
    val dimensions = Codec.VLuint.read(in)
    // non-zero dimensions
    val numCoeffs = Codec.VLuint.read(in)

    var indexValue = 0
    val (indices, values) = (0 until numCoeffs).map { _ =>
      val (delta, coeff) = (Codec.VLuint.read(in), in.readDouble())
      indexValue += delta
      (indexValue, coeff)
    }.unzip
    logger.info(s"Read $numCoeffs dimensions from $dimensions for $uid with intercept $intercept")
    (intercept, Vectors.sparse(dimensions, indices.toArray, values.toArray), uid)
  }
}

/** Paired loader class for IdibonLogisticRegressionModel instances */
class IdibonLogisticRegressionModelLoader
  extends ArchiveLoader[IdibonLogisticRegressionModel] with StrictLogging {


  /** Reloads the object from the Alloy
    *
    * @param engine Engine that houses spark context.
    * @param reader location within Alloy for loading any resources
    *               previous preserved by a call to
    *               { @link com.idibon.ml.common.Archivable#save}
    * @param config archived configuration data returned by a previous
    *               call to { @link com.idibon.ml.common.Archivable#save}
    * @return this object
    */
  def load(engine: Engine, reader: Option[Reader], config: Option[JObject]): IdibonLogisticRegressionModel = {
    implicit val formats = DefaultFormats
    val label = (config.get \ "label" ).extract[String]
    val version = (config.get \ "version" ).extract[String]
    version match {
      case IdibonLogisticRegressionModel.FORMAT_VERSION =>
        logger.info(s"Attemping to load version [v. $version] for '$label'")
      case _ => throw new IOException(s"Unable to load, unhandled version [v. $version] for '$label'")
    }
    val coeffs = reader.get.within("model").resource("coefficients.libsvm")
    val (intercept: Double,
    coefficients: Vector,
    uid: String) = IdibonLogisticRegressionModel.readCodecLibSVM(coeffs)
    coeffs.close()
    val pipeline = CanHazPipeline.loadPipelineIfPresent(engine, reader, config)

    new IdibonLogisticRegressionModel(label,
      new IdibonSparkLogisticRegressionModelWrapper(uid, coefficients, intercept),
      pipeline)
  }
}
