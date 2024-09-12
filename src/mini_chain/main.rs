/*use mini_chain::lib::test_simulation;
#[tokio::main]
async fn main() {
    test_simulation().await.expect("Simulation failed");
}*/
#[tokio::main]
async fn main() {
    test::run_full_node(60, 75).await.expect("Failed to run the full node test");
}
