#include <Arduino.h>

#include <edge/edge.hpp>

using Model = edge::Model<
    edge::InputVector<2>,
    edge::Dense<2, edge::Linear>,
    edge::Dense<1, edge::Linear>>;

static Model model;

static bool nearly_equal(float actual, float expected, float eps = 1.0e-4F) {
    const float diff = actual > expected ? actual - expected : expected - actual;
    return diff <= eps;
}

static bool expect_near(const char* label, float actual, float expected) {
    const bool ok = nearly_equal(actual, expected);
    Serial.print(label);
    Serial.print(": actual=");
    Serial.print(actual, 6);
    Serial.print(" expected=");
    Serial.print(expected, 6);
    Serial.print(" ");
    Serial.println(ok ? "OK" : "FAIL");
    return ok;
}

static bool expect_status(const char* label, edge::Status status) {
    const bool ok = status == edge::Status::Ok;
    Serial.print(label);
    Serial.print(": ");
    Serial.println(edge::status_name(status));
    return ok;
}

static void set_parameters() {
    float* p = model.parameter_data();

    // Dense<2>: row-major weights for two outputs, then two biases.
    p[0] = 0.5F;
    p[1] = -0.25F;
    p[2] = 1.0F;
    p[3] = 0.75F;
    p[4] = 0.1F;
    p[5] = -0.2F;

    // Dense<1>: two weights, then one bias.
    p[6] = 1.5F;
    p[7] = -2.0F;
    p[8] = 0.25F;
}

static bool run_smoke_test() {
    bool ok = true;
    ok = expect_status("model.status", model.status()) && ok;

    Serial.print("parameter_count=");
    Serial.println(static_cast<unsigned long>(Model::parameter_count));
    Serial.print("required_memory=");
    Serial.println(static_cast<unsigned long>(Model::required_memory));
    Serial.print("sizeof_model=");
    Serial.println(static_cast<unsigned long>(sizeof(Model)));
    Serial.print("backward_slot0_count=");
    Serial.println(static_cast<unsigned long>(Model::backward_slot0_count));
    Serial.print("backward_slot1_count=");
    Serial.println(static_cast<unsigned long>(Model::backward_slot1_count));

    set_parameters();

    const float input[2] = {2.0F, -3.0F};
    const float upstream[1] = {2.0F};

    ok = expect_status("zero_grad", model.zero_grad()) && ok;
    ok = expect_status("forward", model.forward(input)) && ok;
    ok = expect_status("backward", model.backward(upstream)) && ok;

    ok = expect_near("output[0]", model.output()[0], 3.925F) && ok;

    const float* g = model.gradient_data();
    ok = expect_near("grad[0]", g[0], 6.0F) && ok;
    ok = expect_near("grad[1]", g[1], -9.0F) && ok;
    ok = expect_near("grad[2]", g[2], -8.0F) && ok;
    ok = expect_near("grad[3]", g[3], 12.0F) && ok;
    ok = expect_near("grad[4]", g[4], 3.0F) && ok;
    ok = expect_near("grad[5]", g[5], -4.0F) && ok;
    ok = expect_near("grad[6]", g[6], 3.7F) && ok;
    ok = expect_near("grad[7]", g[7], -0.9F) && ok;
    ok = expect_near("grad[8]", g[8], 2.0F) && ok;

    return ok;
}

void setup() {
    Serial.begin(9600);
    while (!Serial && millis() < 3000U) {
    }

    Serial.println("EdgeLearning++ Arduino Uno smoke test");
    const bool ok = run_smoke_test();
    Serial.println(ok ? "RESULT: PASS" : "RESULT: FAIL");
}

void loop() {
}
