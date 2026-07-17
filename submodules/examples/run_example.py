from bluehub_submodules import default_parameters, simple_greedy_dispatch, summarize_results


def main() -> None:
    params = default_parameters()
    results = simple_greedy_dispatch(params)
    summary = summarize_results(results)
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()

