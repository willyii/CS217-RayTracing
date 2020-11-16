#ifndef __vec__
#define __vec__

//#include <cassert>
#include <cmath>
#include <iostream>

static const double pi = 4 * std::atan(1.0);

template <class T, int n> struct vec;
template <class T, int n> T dot(const vec<T, n> &u, const vec<T, n> &v);

template <class T, int n> struct vec {
  T x[n];

  vec() { make_zero(); }

  explicit vec(const T &a) {
    // assert(n == 1);
    x[0] = a;
  }

  vec(const T &a, const T &b) {
    // assert(n == 2);
    x[0] = a;
    x[1] = b;
  }

  vec(const T &a, const T &b, const T &c) {
    // assert(n == 3);
    x[0] = a;
    x[1] = b;
    x[2] = c;
  }

  template <class U> explicit vec(const vec<U, n> &v) {
    for (int i = 0; i < n; i++)
      x[i] = (T)v.x[i];
  }

  void make_zero() { fill(0); }

  void fill(T value) {
    for (int i = 0; i < n; i++)
      x[i] = value;
  }

  vec &operator+=(const vec &v) {
    for (int i = 0; i < n; i++)
      x[i] += v.x[i];
    return *this;
  }

  vec &operator-=(const vec &v) {
    for (int i = 0; i < n; i++)
      x[i] -= v.x[i];
    return *this;
  }

  vec &operator*=(const vec &v) {
    for (int i = 0; i < n; i++)
      x[i] *= v.x[i];
    return *this;
  }

  vec &operator/=(const vec &v) {
    for (int i = 0; i < n; i++)
      x[i] /= v.x[i];
    return *this;
  }

  vec &operator*=(const T &c) {
    for (int i = 0; i < n; i++)
      x[i] *= c;
    return *this;
  }

  vec &operator/=(const T &c) {
    for (int i = 0; i < n; i++)
      x[i] /= c;
    return *this;
  }

  vec operator+() const { return *this; }

  vec operator-() const {
    vec r;
    for (int i = 0; i < n; i++)
      r[i] = -x[i];
    return r;
  }

  vec operator+(const vec &v) const {
    vec r;
    for (int i = 0; i < n; i++)
      r[i] = x[i] + v.x[i];
    return r;
  }

  vec operator-(const vec &v) const {
    vec r;
    for (int i = 0; i < n; i++)
      r[i] = x[i] - v.x[i];
    return r;
  }

  vec operator*(const vec &v) const {
    vec r;
    for (int i = 0; i < n; i++)
      r[i] = x[i] * v.x[i];
    return r;
  }

  vec operator/(const vec &v) const {
    vec r;
    for (int i = 0; i < n; i++)
      r[i] = x[i] / v.x[i];
    return r;
  }

  vec operator*(const T &c) const {
    vec r;
    for (int i = 0; i < n; i++)
      r[i] = x[i] * c;
    return r;
  }

  vec operator/(const T &c) const {
    vec r;
    for (int i = 0; i < n; i++)
      r[i] = x[i] / c;
    return r;
  }

  const T &operator[](int i) const { return x[i]; }

  T &operator[](int i) { return x[i]; }

  T magnitude_squared() const { return dot(*this, *this); }

  T magnitude() const { return sqrt(magnitude_squared()); }

  // Be careful to handle the zero vector gracefully
  vec normalized() const {
    T mag = magnitude();
    if (mag)
      return *this / mag;
    vec r;
    r[0] = 1;
    return r;
  };
};

template <class T, int n> vec<T, n> operator*(const T &c, const vec<T, n> &v) {
  return v * c;
}

template <class T, int n> T dot(const vec<T, n> &u, const vec<T, n> &v) {
  T r = 0;
  for (int i = 0; i < n; i++)
    r += u.x[i] * v.x[i];
  return r;
}

template <class T> vec<T, 3> cross(const vec<T, 3> &u, const vec<T, 3> &v) {
  return vec<T, 3>(u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2],
                   u[0] * v[1] - u[1] * v[0]);
}

template <class T, int d>
vec<T, d> componentwise_max(const vec<T, d> &a, const vec<T, d> &b) {
  vec<T, d> r;
  for (int i = 0; i < d; i++)
    r[i] = std::max(a[i], b[i]);
  return r;
}

template <class T, int d>
vec<T, d> componentwise_min(const vec<T, d> &a, const vec<T, d> &b) {
  vec<T, d> r;
  for (int i = 0; i < d; i++)
    r[i] = std::min(a[i], b[i]);
  return r;
}

template <class T, int n>
std::ostream &operator<<(std::ostream &out, const vec<T, n> &u) {
  for (int i = 0; i < n; i++) {
    if (i)
      out << ' ';
    out << u[i];
  }
  return out;
}

template <class T, int n>
std::istream &operator>>(std::istream &in, vec<T, n> &u) {
  for (int i = 0; i < n; i++) {
    in >> u[i];
  }
  return in;
}

typedef vec<double, 2> vec2;
typedef vec<double, 3> vec3;
typedef vec<int, 2> ivec2;
typedef vec<int, 3> ivec3;

#endif
