#pragma once

namespace edge {

enum class Status {
    Ok,
    NullPointer,
    InsufficientArena,
    UnalignedArena,
    InvalidBufferLength,
    InvalidArgument,
    UnsupportedBackend,
    NotInitialized
};

constexpr bool is_ok(Status status) noexcept {
    return status == Status::Ok;
}

constexpr const char* status_name(Status status) noexcept {
    switch (status) {
    case Status::Ok:
        return "Ok";
    case Status::NullPointer:
        return "NullPointer";
    case Status::InsufficientArena:
        return "InsufficientArena";
    case Status::UnalignedArena:
        return "UnalignedArena";
    case Status::InvalidBufferLength:
        return "InvalidBufferLength";
    case Status::InvalidArgument:
        return "InvalidArgument";
    case Status::UnsupportedBackend:
        return "UnsupportedBackend";
    case Status::NotInitialized:
        return "NotInitialized";
    }
    return "Unknown";
}

} // namespace edge

