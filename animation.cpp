#include <iostream>
#include <limits>
#include <string>
#include <assert.h>
#include "animation.hpp"

using namespace std;

static string message;
static size_t value = 0;
static size_t maximum = 0;
static size_t max_line_length = 0;

// Starts a progress indicator in the console
void animation_start(string animation_message, size_t initial,
	size_t animation_maximum) {
	assert(maximum == 0);

	message = animation_message;
	value = initial;
	maximum = animation_maximum;
	if (value >= maximum) {
		animation_finish();
	} else {
		animation_print();
	}
}

// Updates progress
void animation_step() {
	if (maximum == 0) {
		return;
	}

	value++;
	if (value >= maximum) {
		animation_finish();
	} else {
		animation_print();
	}
}

// Removes progress
void animation_finish() {
	cout << string(max_line_length, ' ') << "\r";
	cout.flush();
	value = 0;
	maximum = 0;
	max_line_length = 0;
}

// Displays progress to the user
void animation_print() {
	cout << message << " [" << value << "/" << maximum << "]\r";
	cout.flush();
	max_line_length = message.length() + 4 +
		to_string(value).length() + to_string(maximum).length();
}

