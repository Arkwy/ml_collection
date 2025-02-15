#include <cstdio>

template <typename T> class A {
  protected:
    T t;

  public:
    A(T t) : t(t) {};
    
    void p() { printf("%d", t); }
};

// template <typename T = int> class B {
//   protected:
//     T t;

//   public:
//     B(T t) : t(t+1) {};
    
// };
// // template class A<int>;

// // template <typename T> class B : public A<T> {
// //   private:
// //     T t;
// //   public:
// //     B(T t) : t(t) {};
// // };

// // class C : public A<int> {
// //   private:
// //     int t;
// //   public:
// //     C(int t) : t(t) {};
// //     void p() { printf("%d", t); }
// // };

template<> void A<int>::p() {};


int main() {
    A c(10);
    c.p();
    return 0;
}
