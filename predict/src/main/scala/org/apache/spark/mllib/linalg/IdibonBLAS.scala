package org.apache.spark.mllib.linalg

/**
  * This class extends the linear algebra libraries used internally to Spark so that we can use BLAS operations.
  * The methods are private in Spark -  this makes the method public for use in idiML.
  *
  * @author Michelle Casbon <michelle@idibon.com>
  *
  *
  */
object IdibonBLAS {

  /**
    * Makes this method public for us to access
    * y += a * x
    *
    * Requirements:
    *   - x & y must be the same size
    *   - y must be a DenseVector
    *
    * @param a
    * @param x
    * @param y
    */
  def axpy(a: Double, x: Vector, y: Vector): Unit = {
    BLAS.axpy(a, x, y)
  }

  /**
    * x = a * x
    */
  def scal(a: Double, x: Vector): Unit = {
    BLAS.scal(a, x)
  }

}
