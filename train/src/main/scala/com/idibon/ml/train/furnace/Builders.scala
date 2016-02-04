package com.idibon.ml.train.furnace

import com.idibon.ml.common.Engine
import com.idibon.ml.predict.{Classification, PredictResult}
import org.json4s.ShortTypeHints
import org.json4s.native.Serialization
import org.json4s.native.Serialization.{writePretty}

/**
  * Static object with global defaults for builders.
  */
object BuilderDefaults {
  // LogisticRegression Defaults -- These are the base defaults
  val TOLERANCE = 1e-4
  val REGULARIZATION_PARAMETER = 0.001
  val MAX_ITERATIONS = 100
  val ELASTIC_NET_PARAMETER = 0.9

  // parameter search related defaults -- everything is largely based off of arrays, so
  // if you're doing only one iteration of model params, you take the head of the array of params.
  // that way XVal and SimpleLR can share code easily without separate cases.
  val NUMBER_OF_FOLDS = 10
  val TOLERANCES = Array(TOLERANCE)
  val REGULARIZATION_PARAMETERS = Array(REGULARIZATION_PARAMETER, 0.01, 0.1)
  val ELASTIC_NET_PARAMETERS = Array(ELASTIC_NET_PARAMETER, 1.0)

  val classHints = List(
    classOf[MultiClassLRFurnaceBuilder],
    classOf[SimpleLogisticRegressionBuilder],
    classOf[XValLogisticRegressionBuilder])
  /* This enables us to not do Reflection Foo when trying to create a builder from JSON.
       The only requirement is that has a 'jsonClass' field with one of the names of the classes
       below.*/
  implicit val formats = Serialization.formats(ShortTypeHints(classHints))
}

/**
  * Base FuranceBuilder Trait.
  *
  * Enforces what kind of convention we want to follow.
  *
  * @tparam T The parameterized type of furnace we want to build.
  */
trait FurnaceBuilder[T <: PredictResult] {

  private[furnace] var engine: Engine = null
  /**
    * Each builder needs to have a one of these that takes an engine and
    * returns a furnace
    *
    * @param engine
    * @return
    */
  def build(engine: Engine): Furnace[T]

  /**
    * Creates a pretty printed JSON string.
    *
    * This will be useful for building a tool to output some nice JSON configuration.
    *
    * @return
    */
  override def toString(): String = {
    implicit val formats = BuilderDefaults.formats
    writePretty(this)
  }
}

/*
 Each type of parameter that we could set lives in its own trait.
 Each trait has a `self` to reference to make it easy to create builders
 that work for us.
 Each trait has a an `abstract` declaration of a variable that's needed. It's up to the
 extending class to declare that in its constructor.
 */

/**
  * Use this if you're iterating and want to know when to stop things.
  */
trait HasStochasticOptimizer { self =>
  private[furnace] var maxIterations: Int
  private[furnace] var tolerance: Array[Double]

  def setMaxIterations(maxIterations: Int): self.type = {
    this.maxIterations = maxIterations
    self
  }

  def setTolerance(tolerance: Array[Double]): self.type = {
    this.tolerance = tolerance
    self
  }
}

/**
  * If you can regularize your weights, use this.
  */
trait HasRegularization { self =>

  private[furnace] var regParam: Array[Double]

  def setRegParam(regParam: Array[Double]): self.type = {
    this.regParam = regParam
    self
  }
}

/**
  * If you have an elastic net parameter, use this.
  */
trait HasElasticNet { self =>
  private[furnace] var elasticNetParam: Array[Double]

  def setElasticNetParam(elasticNetParam: Array[Double]): self.type = {
    this.elasticNetParam = elasticNetParam
    self
  }
}

/**
  * If you want to perform XValidation - specify they number of folds.
  */
trait HasXValidation { self =>
  private[furnace] var numFolds: Int

  def setNumberOfFolds(numFolds: Int): self.type = {
    this.numFolds = numFolds
    self
  }
}

/**
  * If you want to split your training set into train & dev to tune parameters.
  */
trait HasTrainingDevSplit { self =>

  private[furnace] var trainingSplit: Double

  def setTrainingSplit(trainingSplit: Double): self.type = {
    this.trainingSplit = trainingSplit
    self
  }
}

/**
  * Builder class to make it easy to create the right SimpleLogisticRegression Furnace.
  *
  * TODO: Doc for each method to explain what is happening.
  *
  * @param maxIterations
  * @param regParam
  * @param tolerance
  * @param elasticNetParam
  */
case class SimpleLogisticRegressionBuilder(private[furnace] var maxIterations: Int = BuilderDefaults.MAX_ITERATIONS,
                                           private[furnace] var regParam: Array[Double] = BuilderDefaults.REGULARIZATION_PARAMETERS,
                                           private[furnace] var tolerance: Array[Double] = BuilderDefaults.TOLERANCES,
                                           private[furnace] var elasticNetParam: Array[Double] = BuilderDefaults.ELASTIC_NET_PARAMETERS)
  extends FurnaceBuilder[Classification] with HasStochasticOptimizer with HasRegularization with HasElasticNet {

  /**
    * Each builder needs to have a one of these that takes an engine and
    * returns a furnace
    *
    * @param engine
    * @return
    */
  override def build(engine: Engine): SimpleLogisticRegression = {
    this.engine = engine
    new SimpleLogisticRegression(this)
  }
}

/**
  * Builder class for creating a MultiClassLRFurnace.
  *
  * @param maxIterations
  * @param regParam
  * @param tolerance
  */
case class MultiClassLRFurnaceBuilder(private[furnace] var maxIterations: Int = BuilderDefaults.MAX_ITERATIONS,
                                      private[furnace] var regParam: Array[Double] = BuilderDefaults.REGULARIZATION_PARAMETERS,
                                      private[furnace] var tolerance: Array[Double] = BuilderDefaults.TOLERANCES)
  extends FurnaceBuilder[Classification] with HasStochasticOptimizer with HasRegularization {

  /**
    * Each builder needs to have a one of these that takes an engine and
    * returns a furnace
    *
    * @param engine
    * @return
    */
  override def build(engine: Engine): MultiClassLRFurnace = {
    this.engine = engine
    new MultiClassLRFurnace(this)
  }
}

/**
  * Builder class to make it easy to create the right XValLR Furnace.
  *
  * TODO: Doc for each method to explain what is happening.
  *
  */
case class XValLogisticRegressionBuilder(private[furnace] var maxIterations: Int = BuilderDefaults.MAX_ITERATIONS,
                                         private[furnace] var regParam: Array[Double] = BuilderDefaults.REGULARIZATION_PARAMETERS,
                                         private[furnace] var tolerance: Array[Double] = BuilderDefaults.TOLERANCES,
                                         private[furnace] var elasticNetParam: Array[Double] = BuilderDefaults.ELASTIC_NET_PARAMETERS,
                                         private[furnace] var numFolds: Int = BuilderDefaults.NUMBER_OF_FOLDS)
  extends FurnaceBuilder[Classification] with HasStochasticOptimizer with HasRegularization with HasElasticNet{


  override def build(engine: Engine): XValLogisticRegression = {
    this.engine = engine
    new XValLogisticRegression(this)
  }
}
