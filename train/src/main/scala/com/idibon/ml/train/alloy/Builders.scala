package com.idibon.ml.train.alloy

import com.idibon.ml.common.Engine
import com.idibon.ml.predict.{PredictResult, Classification}
import com.idibon.ml.train.datagenerator.{MultiClassDataFrameGeneratorBuilder, KClassDataFrameGeneratorBuilder, SparkDataGeneratorBuilder}
import com.idibon.ml.train.furnace.{MultiClassLRFurnaceBuilder, XValLogisticRegressionBuilder, FurnaceBuilder}
import org.json4s.ShortTypeHints
import org.json4s.native.Serialization
import org.json4s.native.Serialization._

/**
  * Static object to house global defaults for Alloy Trainer builders.
  */
object BuilderDefaults {
  val classHints = List(classOf[KClass1FPBuilder], classOf[MultiClass1FPBuilder])
  // we need to connect all the different possible classes underneath so we can
  // create a single JSON config that gets split into the respective builders.
  implicit val formats = Serialization.formats(ShortTypeHints(classHints ++
    com.idibon.ml.train.datagenerator.BuilderDefaults.classHints ++
    com.idibon.ml.train.furnace.BuilderDefaults.classHints
  ))
}

/**
  * Trait that all alloy builders need to extend.
  */
trait AlloyTrainerBuilder {

  /**
    * Returns the alloy trainer.
 *
    * @param engine
    * @return
    */
  def build(engine: Engine): AlloyTrainer

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

/**
  * Builder for creating the K Class 1 FP trainer.
 *
  * @param dataGenBuilder The builder that produce the data generator for input to fit models.
  * @param furnaceBuilder The builder that produces the class to fit models.
  */
case class KClass1FPBuilder(private[alloy] var dataGenBuilder: SparkDataGeneratorBuilder = new KClassDataFrameGeneratorBuilder(),
                            private[alloy] var furnaceBuilder: FurnaceBuilder[Classification] = new XValLogisticRegressionBuilder())
  extends AlloyTrainerBuilder {
  private[alloy] var engine: Engine = null

  override def build(engine: Engine): KClass1FP = {
    this.engine = engine
    new KClass1FP(this)
  }
}

/**
  * Builder for creating the Multi-class 1 FP trainer.
 *
  * @param dataGenBuilder The builder that produce the data generator for input to fit models.
  * @param furnaceBuilder The builder that produces the class to fit models.
  */
case class MultiClass1FPBuilder(private[alloy] var dataGenBuilder: SparkDataGeneratorBuilder = new MultiClassDataFrameGeneratorBuilder(),
                                private[alloy] var furnaceBuilder: FurnaceBuilder[Classification] = new MultiClassLRFurnaceBuilder())
  extends AlloyTrainerBuilder {
  private[alloy] var engine: Engine = null

  override def build(engine: Engine): MultiClass1FP = {
    this.engine = engine
    new MultiClass1FP(this)
  }
}
