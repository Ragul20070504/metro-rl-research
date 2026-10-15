"""Shared infrastructure: paths, configuration provenance, seeding, manifests, logging.

Nothing scientific lives here. These modules exist so that the scientific modules cannot
accidentally violate the protocol -- by hard-coding a path, by using a parameter with no
provenance, by breaking the common-random-numbers guarantee, or by overwriting a stored run.
"""
