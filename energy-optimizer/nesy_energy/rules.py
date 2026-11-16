"""Default plant rule book. Add site-specific rules through Plant.extra_rules."""

DEFAULT_RULES = """
[R1] Two high-inrush machines on the same feeder must never run together (breaker protection).
conflict(A,B) :- high_inrush(A), high_inrush(B), feeder(A,F), feeder(B,F), A != B.

[R2] A process that needs an operator cannot run in unstaffed hours.
forbidden(T,H) :- needs_operator(T), unstaffed(H).

[R3] Noisy processes cannot run during community quiet hours.
forbidden(T,H) :- noisy(T), quiet_hour(H).

[R4] A process must finish before the downstream process it feeds may start.
precedes(A,B) :- feeds(A,B).

[R5] Process order is transitive along the production chain.
precedes(A,C) :- precedes(A,B), precedes(B,C).

[R6] A machine under maintenance lock-out cannot run.
forbidden(T,H) :- maintenance(T,H).

[R7] Two processes cannot use the same exclusive resource at once.
conflict(A,B) :- uses(A,R), uses(B,R), exclusive(R), A != B.
"""
