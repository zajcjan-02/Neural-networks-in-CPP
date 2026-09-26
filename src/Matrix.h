#pragma once

#include <vector>
#include <cmath>
#include <random>
#include <cstddef>
#include <stdexcept>
#include <type_traits>

template<typename T>
class Matrix {
	private:
		std::size_t rows_;
		std::size_t cols_;
		std::vector<T> data_;

	public:
		Matrix(std::size_t rows, std::size_t cols): rows_(rows), cols_(cols), data_(rows * cols) {}
		Matrix(std::size_t rows, std::size_t cols, const std::vector<T>& data): rows_(rows), cols_(cols), data_(data) {
			if (data.size() != rows * cols) {
				throw std::invalid_argument("Matrix data size does not match dimensions");
			}
		}
		class Row {
			private:
				T* ptr_;

			public:
				explicit Row(T* ptr) : ptr_(ptr) {}

				T& operator[](std::size_t col) {
					return ptr_[col];
				}
		};

		class ConstRow {
			private:
				const T* ptr_;

			public:
				explicit ConstRow(const T* ptr) : ptr_(ptr) {}

				const T& operator[](std::size_t col) const {
					return ptr_[col];
				}
		};

		Row operator[](std::size_t row) {
			return Row(data_.data() + row * cols_);
		}

		ConstRow operator[](std::size_t row) const {
			return ConstRow(data_.data() + row * cols_);
		}

		[[nodiscard]] std::size_t rows() const {
			return rows_;
		}

		[[nodiscard]] std::size_t cols() const {
			return cols_;
		}

		[[nodiscard]] std::size_t size() const {
			return data_.size();
		}

		T& operator()(std::size_t row, std::size_t col) {
			return data_[row * cols_ + col];
		}

		const T& operator()(std::size_t row, std::size_t col) const {
			return data_[row * cols_ + col];
		}

		void xavier_initialization() {
			static_assert(std::is_floating_point_v<T>,
				"Xavier initialization requires a floating-point Matrix");

			const T limit = std::sqrt(
				static_cast<T>(6) /
				static_cast<T>(rows_ + cols_)
			);

			static std::random_device rd;
			static std::mt19937 gen(rd());

			std::uniform_real_distribution<T> dist(-limit, limit);

			for (T& w : data_) {
				w = dist(gen);
			}
		}
	};