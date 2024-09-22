def get_figure_plots(
    env,
    states,
    energies,
    fidelity,
    logger,
    title,
    key,
    use_context,
):
    if (hasattr(env, "env") and hasattr(env.env, "plot_samples_frequency")) or (
        hasattr(env, "env") == False and hasattr(env, "plot_samples_frequency")
    ):
        fig = env.plot_samples_frequency(
            states,
            title=title + "_Samples_Frequency",
            rescale=env.rescale,
        )
        logger.log_figure(key + "_samples_frequency", fig, use_context=use_context)
    if (hasattr(env, "env") and hasattr(env.env, "plot_reward_distribution")) or (
        hasattr(env, "env") == False and hasattr(env, "plot_reward_distribution")
    ):
        fig = env.plot_reward_distribution(
            scores=energies,
            fidelity=fidelity,
            title=title + "_Reward_Distribution",
        )
        logger.log_figure(key + "_reward_distribution", fig, use_context=use_context)
