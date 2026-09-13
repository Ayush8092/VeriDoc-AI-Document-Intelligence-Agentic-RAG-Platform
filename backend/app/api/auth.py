"""POST /auth/register, POST /auth/login, GET /auth/me.

See app/security/auth.py for the password-hashing/token machinery this
router wraps. Errors are deliberately generic ("Invalid email or
password.") on login regardless of whether the email doesn't exist or
the password is wrong — distinguishing the two would let an attacker
enumerate registered emails (spec section 11, "avoid leaking internal
information").
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import IntegrityError

from app.core.config import Settings, get_settings
from app.db.models import User
from app.db.session import init_db, session_scope
from app.schemas.auth import LoginRequest, RegisterRequest, TokenResponse, UserOut
from app.security.auth import create_access_token, get_current_user_required, hash_password, verify_password

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def register(payload: RegisterRequest, settings: Settings = Depends(get_settings)) -> TokenResponse:
    init_db(settings)
    with session_scope(settings) as session:
        existing = session.query(User).filter(User.email == payload.email.lower()).one_or_none()
        if existing is not None:
            # Same generic-error principle as login (see module
            # docstring) — but registration inherently confirms an email
            # is taken either way (that's the nature of "sign up"), so a
            # specific 409 is fine here, unlike login.
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="An account with this email already exists.")

        user = User(email=payload.email.lower(), hashed_password=hash_password(payload.password))
        session.add(user)
        try:
            session.flush()
        except IntegrityError:
            # Race: two concurrent registrations for the same email.
            session.rollback()
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT, detail="An account with this email already exists."
            ) from None

        token = create_access_token(user, settings)
        return TokenResponse(access_token=token, expires_in_minutes=settings.jwt_expire_minutes)


@router.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest, settings: Settings = Depends(get_settings)) -> TokenResponse:
    init_db(settings)
    with session_scope(settings) as session:
        user = session.query(User).filter(User.email == payload.email.lower()).one_or_none()
        if user is None or not user.is_active or not verify_password(payload.password, user.hashed_password):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password.")
        token = create_access_token(user, settings)
        return TokenResponse(access_token=token, expires_in_minutes=settings.jwt_expire_minutes)


@router.get("/me", response_model=UserOut)
def me(current_user: User = Depends(get_current_user_required)) -> UserOut:
    return UserOut(**current_user.to_dict())
