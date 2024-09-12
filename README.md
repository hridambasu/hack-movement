# Mini Chain Take Home Technical
- **Duration:** 1:15 - 2:00
- **Overview:** Answer technical and conceptual questions and patch a couple holes in this small blockchain.
- We fully expect you to use LLM support! We're more interested in what your answers reveal about your approach to programming.
- This technical comes in **FIVE PARTS**:
    - Some warm-up questions. Called out in code and in this document with [W#]
    - Programming tasks. Called out in code and in this document with [P#].
    - Simulation tasks. Called out in code and in this document with [S#].
        - These should be achieved by your completion of the programming tasks, but test e2e performance of the system.
    - Rust specific questions. Called out in code and in this document with [R#].
    - Conceptual questions. Called out in code and in this document with [C#].

## The Mini Chain
Mini Chain is a small proof of work blockchain. In its current state, it lacks both block verification and chain selection algorithms. Your programmatic task will be to add these in.

### Architecture 
The basic architecture of MiniChain is defined in three traits: `Proposer`, `Miner`, and `Verifier`.

#### `Proposer`
The proposer reads transactions from the mempool and proposes blocks.

```rust
#[async_trait]
pub trait Proposer {

    // Builds a block from the mempool
    async fn build_block(&self) -> Result<Block, String>;

    // Sends a block to the network for mining
    async fn send_proposed_block(&self, block : Block)->Result<(), String>;

    // Propose a block to the network and return the block proposed
    async fn propose_next_block(&self) -> Result<Block, String> {
        let block = self.build_block().await?;
        self.send_proposed_block(block.clone()).await?;
        Ok(block)
    }

}
```

#### `Miner`
The miner receives proposed blocks and submits them for verification.
```rust
#[async_trait]
pub trait Miner {

    // Receives a block over the network and attempts to mine
    async fn receive_proposed_block(&self) -> Result<Block, String>;

    // Sends a block to the network for validation
    async fn send_mined_block(&self, block : Block) -> Result<(), String>;

    // Mines a block and sends it to the network
    async fn mine_next_block(&self) -> Result<Block, String> {
        let block = self.receive_proposed_block().await?;
        // default implementation does no mining
        self.send_mined_block(block.clone()).await?;
        Ok(block)
    }

}
```

#### `Verifier`
The verifier receives mined blocks, verifiers them, and updates its representation of the chain.
```rust
#[async_trait]
pub trait Verifier {

    // Receives a block over the network and attempts to verify
    async fn receive_mined_block(&self) -> Result<Block, String>;

    // Verifies a block
    async fn verify_block(&self, block: &Block) -> Result<bool, String>;

    // Synchronizes the chain
    async fn synchronize_chain(&self, block: Block) -> Result<(), String>;

    // Verifies the next block
    async fn verify_next_block(&self) -> Result<(), String> {
        let block = self.receive_mined_block().await?;
        let valid = self.verify_block(&block).await?;
        if valid {
            self.synchronize_chain(block).await?;
        }
        Ok(())
    }

}
```

#### `FullNode`
A `FullNode` combines the `Proposer`, `Miner`, `Verifier`, and more to provide a complete utility for running the blockchain.
```rust
#[async_trait]
pub trait FullNode 
    : Initializer + 
    ProposerController + MinerController + VerifierController + 
    MempoolController + MempoolServiceController +
    BlockchainController + Synchronizer + SynchronizerSerivce {

    /// Runs the full node loop
    async fn run_full_node(&self) -> Result<(), String> {
        self.initialize().await?;
        loop {
            let error = try_join!(
                self.run_proposer(),
                self.run_miner(),
                self.run_verifier(),
                self.run_mempool(),
                self.run_blockchain(),
                self.run_mempool_service(),
                self.run_synchronizer_service()
            ).expect_err("One of the controllers stopped without error.");

            println!("Error: {}", error);
        }

    }

}
```

### Helpful Code
Generally speaking you'll find the following useful

#### `BlockchainOperations::pretty_print_tree(&self)`
This will show you a marked up version of the block tree. It annotates the main chain, the chain tip, leaves, block height, etc. It will produce output like the below.
```
Full Node #1
└── (0) Block: f1534392279bddbf 
    └── (1) Block: 001c9f15b438d405 
        ├── (2) Block: 00b53870fa564e32 < TIP
        ├── (2) Block: 002fcec250aa0e43
        ├── (2) Block: 000815e5bd5161bf
        └── (2) Block: 00bf6f30ed06136f
```

#### Tuning `fast_chain` default for `BlockchainMetadata`
`BlockchainMetadata::fast_chain` is supposed to contain a set of parameters for the chain that enable it to be run by several full nodes on the same device. Tuning these parameters will help you produce different outcomes and ideally pass the simulation tests.
```rust
impl BlockchainMetadata {
    pub fn fast_chain() -> Self {
        BlockchainMetadata {
            query_slots: 4,
            slot_secs: 2,
            fork_resolution_slots : 2,
            block_size: 128,
            maximum_proposer_time: 1,
            mempool_reentrancy_secs: 2,
            transaction_expiry_secs: 8,
            difficulty: 2,
            proposer_wait_ms : 50
        }
    }
}
```

#### The tests
Generally speaking the tests which are not simulations are quite useful. Most of them should not be affected by your work and serve more as indications of how various systems work. However, the tests in `mini_chain/chain.rs` will likely be very relevant to confirming the desired behavior.

### The Assessment
This technical comes in five parts as mentioned above.

#### The Warmup
A selection of short exercises to get your brain going and get familiarized with the source.

##### [W1]
This function to check whether the chain has been resolved (leaves pruned) is unimplemented. Fix that. 

**HINT:** Count the leaves!
```rust
async fn is_resolved(&self) -> bool {
    // Check if there's only a single leaf
    // TODO: [W1] this is unimplemented.
    false
}
```

Solution:
```rust
    async fn is_resolved(&self) -> bool {
        // Check if there's only a single leaf
        self.leaves.len() == 1
    }
```

##### [W2]
This implementation of a method to generate a random Argument could blow the stack. Fix it.
```rust
impl Argument {
    pub fn random<R: Rng>(rng: &mut R, depth: u64) -> Self {

        // TODO [W2]: This is wrong. Prevent this from blowing the stack.
        let partition = if depth > 0 { 9 } else { 8 }; 

        match rng.gen_range(0..partition) {
            0 => Argument::U8(rng.gen()),
            1 => Argument::U16(rng.gen()),
            2 => Argument::U32(rng.gen()),
            3 => Argument::U64(rng.gen()),
            4 => Argument::U128(rng.gen()),
            5 => Argument::Address(Address::random(rng)),
            6 => Argument::Signer(Signer(Address::random(rng))),
            7 => {
                let len = rng.gen_range(1..8);
                let args: Vec<Argument> = (0..len).map(|_| Argument::random(rng, depth - 1)).collect();
                Argument::Vector(args)
            },
            8 => {
                let len = rng.gen_range(1..8);
                let args: Vec<Argument> = (0..len).map(|_| Argument::random(rng, depth - 1)).collect();
                Argument::Struct(args)
            },
            _ => unreachable!(),
        }
    }
}
```

Solution:
```rust
    pub fn random<R: Rng>(rng: &mut R, depth: u64) -> Argument {
        // TODO [W2]: This is wrong. Prevent this from blowing the stack.
        let partition = if depth > 0 { 9 } else { 8 };
    
        match rng.gen_range(0..partition) {
            0 => Argument::U8(rng.gen()),
            1 => Argument::U16(rng.gen()),
            2 => Argument::U32(rng.gen()),
            3 => Argument::U64(rng.gen()),
            4 => Argument::U128(rng.gen()),
            5 => Argument::Address(Address::random(rng)),
            6 => Argument::Signer(Signer(Address::random(rng))),
            7 => {
                let len = rng.gen_range(1..8);
                let args: Vec<Argument> = (0..len)
                    .map(|_| Argument::random(rng, if depth > 0 { depth - 1 } else { 0 }))
                    .collect();
                Argument::Vector(args)
            }
            8 => {
                let len = rng.gen_range(1..8);
                let args: Vec<Argument> = (0..len)
                    .map(|_| Argument::random(rng, if depth > 0 { depth - 1 } else { 0 }))
                    .collect();
                Argument::Struct(args)
            }
            _ => unreachable!(),
        }
    }
```
#### Programming
Programming tasks.

##### [P1]
We have not implemented verification for proof of work. Implement this using a leading-zeros alg.
```rust
// TODO [P1]: Implement proof of work verification for a difficulty (leading zeros).
pub fn verify_proof_of_work(&self, difficulty: usize) -> bool {
    false
}
```

Solution:
```rust
    pub fn verify_proof_of_work(&self, difficulty: usize) -> bool {
        let hash_with_nonce = format!("{}{:x}", self.calculate_hash(), self.nonce);
        let hash = Sha256::digest(hash_with_nonce.as_bytes());

        // Check if the hash has the required number of leading zeros
        // Convert the GenericArray to a slice and check if it has the required leading zeros
        let hash_slice = &hash[..];
        hash_slice.starts_with(&[0; 1].repeat(difficulty as usize))
    }
```

##### [P2]
Implement your favorite chain selection alg. Remember, we store a block tree as an adjacency matrix in a `HashSet`. 

**HINT:** Longest chain is likely the most reasonable alg to implement.
```rust
async fn get_main_chain(&self) -> Result<Vec<Block>, String> {

    // TODO [P2]: Implement your preffered chain selection algorithm here
    Ok(Vec::new())
    
}
```

Solution:
```rust
    async fn get_main_chain(&self) -> Result<Vec<Block>, String> {
        let mut main_chain = Vec::new();
        let mut visited = HashSet::new();
    
        // Iterate over all blocks and find the block with the maximum height
        let max_height_block = self
            .blocks
            .iter()
            .filter(|(block_hash, _)| !visited.contains(*block_hash))
            .max_by_key(|(block_hash, _)| self.calculate_chain_height(*block_hash, &visited));
    
        if let Some((start_block, _)) = max_height_block {
            // Reconstruct the main chain starting from the block with maximum height
            self.construct_chain_recursive(start_block, &mut main_chain, &mut visited);
        }
    
        Ok(main_chain)
    }
```

#### Entering the Simulation
Make the simulations pass the benchmarks.
##### [S1]
Generate passing output from the full node simulation in `mini_chain/node.rs`. Provide a screenshot of your final chain and an intermediate chain below.

Solution:
The images are located in the path:

![Final Chain](/images/sim1/final_chain.png)
![Intermediate Chain](/images/sim1/intermediate_chain.png)

##### [S2]
Generate passing output from the network simulation in `lib.rs`. Provide a screenshot of your final chain and an intermediate chain below.

Solution:
The images are located in the path:

![Final Chain](/images/sim2/final_chain_2.png)
![Intermediate Chain](/images/sim2/intermediate_chain_2.png)

#### Rustacean Station
Answer these Rust-oriented questions. Answer in the space below each question.

##### [R1]
Why is sharing a reference to a `Receiver` potentially dangerous? 

**HINT:** How is a receiver consumed?

```rust
#[derive(Debug, Default, Clone)] // TODO [R1]: Why is it potentially dangerous to derive clone?
pub struct Blockchain {
    blocks: HashMap<String, (Block, HashSet<String>)>,
    leaves: HashSet<String>,
    slots: BTreeMap<u64, HashSet<String>>,
    metadata : Arc<BlockchainMetadata>, // blockchain does not mutate its own metadata
}
```

Solution: Sharing a reference to a Receiver in Rust can be potentially dangerous because Receiver itself is not Sync. The Sync trait in Rust is used to indicate that it's safe to share references between threads. Since Receiver is not Sync, sharing a reference to it among multiple threads can lead to undefined behavior, data races, and memory unsafety. The primary reason for this is that Receiver may have internal mutable state to keep track of the messages it has received or other bookkeeping information. Sharing a non-Sync type across threads without proper synchronization mechanisms can result in race conditions, where multiple threads concurrently access and modify the internal state of the Receiver, leading to unpredictable behavior.
To safely share a Receiver among multiple threads, we can use synchronization primitives like Arc<Mutex<Receiver<T>>> or Arc<RwLock<Receiver<T>>>. These wrappers ensure that access to the Receiver is properly synchronized, preventing data races and ensuring safe concurrent access.

The #[derive(Clone)] attribute for deriving the Clone trait in Rust is not inherently dangerous, but it can introduce potential risks depending on the properties of the types involved. In the case of your Blockchain struct, deriving Clone might not be problematic. Let's break down the potential considerations:

Deep or Shallow Copy: The Clone trait creates a copy of the entire structure. If your Block type and HashSet<String> inside the HashMap are also Clone, then the derived Clone implementation for Blockchain will perform a deep copy, copying each element inside the struct. If they are not Clone or have expensive clone implementations, it might result in unnecessary overhead.

Arc Cloning: The Arc<BlockchainMetadata> is an Arc (atomic reference counting) smart pointer, and cloning it increments the reference count, not creating a deep copy of the underlying data. This is generally efficient and safe. However, if the metadata contains mutable state, you might want to ensure that the mutation is properly synchronized.

Performance Concerns: In certain cases, manual implementation of the Clone trait might be more performant or allow for more control over the cloning process, especially if your struct contains non-Clone types or types where cloning is expensive.
We can manually implement Clone for the Blockchain struct in the following manner:
```rust
impl Clone for Blockchain {
    fn clone(&self) -> Self {
        Blockchain {
            blocks: self.blocks.clone(),
            leaves: self.leaves.clone(),
            slots: self.slots.clone(),
            metadata: self.metadata.clone(),
        }
    }
}
```
In summary, deriving Clone is not inherently dangerous, but we should be aware of how the cloning process behaves for the types involved in your struct. If deep copying is expensive or if we need more control over the cloning process, manual implementation might be preferred.

##### [R2]
Compare and contrast the approaches to conditional traits used in `mini_chain/node.rs` and `mini_chain/mempool.rs`.

Solution:
Let us denote the approaches to conditional traits used in `mini_chain/node.rs` as Approach 1 and that in `mini_chain/mempool.rs` as
Approach 2.

Approach 1:

1. The trait SynchronizerService is defined as an extension of the Synchronizer trait.
2. The async_trait macro is used to declare an asynchronous trait.
3. The trait includes an asynchronous function run_synchronizer_service, which runs a loop that calls check_for_next_block and yields the processor.
4. This approach focuses on a single trait, SynchronizerService, extending another trait, Synchronizer. It includes a single asynchronous function related to synchronizing.
5. This approach is suitable for scenarios where a specific trait extension is needed, and the implementation revolves around a single asynchronous function (run_synchronizer_service).

Approach 2:

1. The conditional trait is implemented for a generic type T, which must implement several other traits (QueueInFlightMempool, MempoolBlockchainOperations, MempoolMetadataOperations, etc.).
2. The async_trait macro is used to declare an asynchronous trait implementation.
3. The trait implementation includes multiple asynchronous functions related to mempool operations, such as checking transaction acceptability, pushing transactions, popping transactions, and ticking the mempool.
4. This approach implements the MempoolOperations trait for a generic type T, requiring T to implement multiple other traits. It includes several asynchronous functions related to mempool operations.
5. This approach is suitable for scenarios where a set of related asynchronous functions is implemented for a generic type, assuming it satisfies certain trait requirements.

Here are the commonalities between the two approaches:

1. Both approaches use the async_trait macro to declare asynchronous traits or trait implementations.
2. Both involve asynchronous functions.

##### [R3]
Identify the pattern that is forced by the definition of the Verifier trait--assuming the Verifier would in fact mutate state.
```rust
// TODO [R3]: Identify the pattern that is forced by this trait--assuming the Verifier would in fact mutate state.
#[async_trait]
pub trait Verifier {

    // Receives a block over the network and attempts to verify
    async fn receive_mined_block(&self) -> Result<Block, String>;

    // Verifies a block
    async fn verify_block(&self, block: &Block) -> Result<bool, String>;

    // Synchronizes the chain
    async fn synchronize_chain(&self, block: Block) -> Result<(), String>;

    // Verifies the next block
    async fn verify_next_block(&self) -> Result<(), String> {
        let block = self.receive_mined_block().await?;
        let valid = self.verify_block(&block).await?;
        if valid {
            self.synchronize_chain(block).await?;
        }
        Ok(())
    }

}
```
Solution:
The pattern forced by the Verifier trait, assuming it mutates state, is an asynchronous chain of operations for receiving, verifying, and synchronizing blocks in a blockchain-like system.

This is a list of steps demonstrating how the pattern works:

1. The receive_mined_block function asynchronously receives a block over the network.
2. The verify_block function asynchronously verifies the received block.
3. The synchronize_chain function asynchronously synchronizes the chain with the received block.
4. The verify_next_block function is a default implementation that combines the above operations in a sequence:
    i. It calls receive_mined_block to get the next block.
    ii. It then verifies the received block using verify_block.
    iii. If the block is valid, it synchronizes the chain with the received block using synchronize_chain.

This pattern ensures that blocks are received, verified, and synchronized in the correct order, maintaining the integrity and consistency of the blockchain. It also allows for easy extension and customization by implementing the Verifier trait for different types of verifiers with potentially different verification and synchronization strategies.


##### [R4]
Explain how the `broadcast_message` function works in our `Network` simulator. Reference the functionality of `Sender` and `Receiver`.

```rust
// TODO [R4]: Explain how this broadcast_message function works
pub async fn broadcast_message<T: Clone + Send + Debug + 'static>(
    &self,
    ingress_receiver: Arc<RwLock<Receiver<T>>>,
    egress_senders: &[Sender<T>],
) -> Result<(), String> {
    loop {

        let message = {
            let mut receiver = ingress_receiver.write().await;
            receiver.recv().await.unwrap()
        };

        let mut broadcasts = Vec::new();
        for sender in egress_senders {
            let cloned_message = message.clone();
            broadcasts.push(async {
                if !self.metadata.read().await.should_drop() {
                    self.metadata.read().await.introduce_latency().await;
                    sender.send(cloned_message).await.unwrap();
                }
            });
        }
        futures::future::join_all(broadcasts).await;
    }
}
```
Solution:
The broadcast_message function in the Network simulator is responsible for broadcasting a message received from an ingress receiver to multiple egress senders while taking into account network metadata such as latency and message dropping.

This is the control flow of the function:

1. It takes an ingress receiver and a slice of egress senders as parameters.
2. It enters an infinite loop to continuously broadcast messages.
3. Within the loop:
    i. It awaits for a message to be received from the ingress receiver. This is done by acquiring a write lock on the ingress receiver and calling recv().await to asynchronously receive a message. The received message is then unwrapped.
    ii. It initializes an empty vector broadcasts to hold asynchronous tasks for broadcasting the message to each egress sender.
    iii. It iterates over each egress sender in the provided slice.
    iv. For each egress sender:
        a. It clones the message to be sent since each egress sender may consume it independently.
        b. It constructs an asynchronous task that sends the cloned message through the egress sender after checking whether the network should drop the message or introduce latency based on metadata.
        c. It pushes this asynchronous task into the broadcasts vector.
        d. It uses futures::future::join_all to asynchronously wait for all the broadcasting tasks in the broadcasts vector to complete. This ensures that all messages are sent concurrently to their respective egress senders.

Thus the broadcast_message function efficiently broadcasts a message received from an ingress receiver to multiple egress senders, taking advantage of asynchronous programming to handle each sender concurrently. It also incorporates network metadata such as message dropping and latency, ensuring realistic simulation behavior.

#### Blockchainceptual
Conceptual questions about the blockchain. Answer in the space below each question.

##### [C1]
What effect should we expect changing the number of transactions in a block to have on our Blockchain? Would it help or hurt temporary forking?

Solution:
Changing the number of transactions in a block can have several effects on a blockchain system:

1. Throughput: Increasing the number of transactions in a block generally increases the throughput of the blockchain. This means more transactions can be processed and confirmed in a given time period, which is beneficial for scalability.
2. Confirmation Time: With more transactions per block, it might take longer to validate and confirm each block. This could potentially increase the time it takes for a transaction to be confirmed and included in a block.
3. Blockchain Size: Larger blocks result in larger blockchain size over time, as more data is stored in each block. This can affect storage requirements for network participants.
4. Network Latency: Larger blocks may increase network latency as they take longer to propagate across the network. This could potentially lead to slower block propagation times and potentially increase the risk of temporary forks.

Regarding temporary forking, increasing the number of transactions in a block could potentially have both positive and negative effects:

Positive Effects: Larger blocks may reduce the frequency of temporary forks because they contain more transactions, which means there's a higher probability that conflicting transactions are included in the same block, reducing the chances of divergent chains.

Negative Effects: On the other hand, larger blocks may increase the propagation time of blocks across the network, which could increase the likelihood of temporary forks due to delayed propagation and validation of blocks.

##### [C2]
What would happen if we did not initialize the chain representations with a genesis block?

Solution:
If a blockchain representation is not initialized with a genesis block, several significant issues could arise:

1. Invalid State: Without a genesis block, the blockchain would be in an invalid state because every blockchain needs to start with a genesis block. The genesis block serves as the initial point of reference from which subsequent blocks are built upon.
2. Chain Integrity: The absence of a genesis block means there is no starting point for the blockchain. As a result, there would be no way to verify the integrity and validity of subsequent blocks since there would be no known state to compare against.
3. Consensus Mechanism: Many consensus mechanisms rely on the existence of a genesis block to establish the initial state of the blockchain and to define the rules for block creation and validation. Without a genesis block, the consensus mechanism would not have a basis for determining the validity of new blocks.
4. Forking and Consistency: If different nodes in the network attempt to build a blockchain without a genesis block, they may diverge into separate chains, leading to inconsistency and potential forking. Nodes may not agree on the starting point of the blockchain, leading to disagreements on subsequent blocks and transactions.
5. Security Risks: Without a properly initialized blockchain, there are security risks such as the potential for malicious actors to inject invalid or malicious blocks into the chain, leading to various forms of attacks such as double spending or invalid data insertion.

So, initializing the chain representation with a genesis block is a fundamental requirement for the proper functioning and integrity of a blockchain. It provides a starting point, defines the rules for block creation, and ensures consistency and security within the network.

##### [C3]
One of our simulation benchmarks was the chains' edit score. What qualities of our network does this best measure (if any)? Is there a better way to measure something similar.
```rust
fn chains_edit_score(main_chains : Vec<Vec<String>>) -> f64 {

    let mut score = 0.0;
    let mut max_score = 0.0;

    for (i, chain_a) in main_chains.iter().enumerate() {
        let length_a = chain_a.len();
        for (j, chain_b) in main_chains.iter().enumerate() {
            let length_b = chain_b.len();
            if i != j {
                score += levenshtein_distance_vec(chain_a, chain_b) as f64;
                max_score += (length_a + length_b) as f64;
            }
        }
    }

    1.0 - (score/max_score)

}
```

Solution:
The qualities of the network that this function may best measure include:

1. Chain Similarity: It measures how similar or dissimilar the chains are to each other in the network. Higher similarity scores suggest that the chains are more closely related, possibly indicating better network consistency.
2. Network Resilience: It indirectly measures the network's resilience to forks or divergent chains. Lower scores indicate that the chains are more divergent, potentially indicating a less resilient network.

However, there may be better ways to measure similar qualities in a blockchain network, depending on the specific goals and requirements:

1. Consensus Agreement: Instead of measuring similarity based on edit distances, directly measuring the level of agreement among nodes in the network regarding the current state of the blockchain could provide a more accurate measure of network consensus.
2. Fork Detection: Developing metrics specifically designed to detect and quantify the occurrence and impact of forks in the network could provide insights into network resilience and stability.
3. Blockchain Integrity: Metrics focusing on the integrity and consistency of the blockchain, such as the number of orphaned blocks or the rate of block reorganizations, could provide valuable information about the health of the network.

To measure "Consensus Agreement" in order to assess chain similarity, you could consider the following approach:

1. Define Consensus Agreement: Firstly, we need to define what we mean by "Consensus Agreement." It could refer to the level of similarity or overlap between the blocks in different chains.

2. Select Consensus Metrics: Then we need to choose specific metrics or criteria that indicate consensus agreement such as:
    a. Block Content: Compare the content of blocks, such as transactions, timestamps, and other metadata.
    b. Block Hashes: Despite the limitation mentioned, comparing block hashes could still provide some insight, especially if we exclude the previous block from the hash as you suggested. This would essentially measure agreement at a shallow depth, but it could still be informative.
    c. Proof of Work/Consensus Mechanism: Evaluate whether the chains share the same consensus mechanism (e.g., proof of work) and assess the level of agreement in terms of computational effort expended.
    d. Fork Density: Measure the frequency of forks or divergences in the chains and assess how quickly consensus is reached after forks.

3. Data Preparation: Ensure that the chains that are being compared are aligned properly for analysis. This might involve synchronizing the chains to a common starting point or ensuring that corresponding blocks are aligned correctly.

4. Compute Consensus Agreement Score: Based on the selected metrics, compute a consensus agreement score for the chains. This could involve calculating a weighted similarity score or agreement percentage based on the chosen metrics for some choice of weights between the different metrics.

5. Normalize and Interpret: Normalize the consensus agreement score if necessary to ensure comparability across different scenarios or datasets. Then, interpret the score to assess the level of similarity or consensus agreement between the chains.

6. Validation and Sensitivity Analysis: Validate the chosen metrics and the computed consensus agreement score by comparing with known scenarios or conducting sensitivity analyses to ensure robustness.

##### [C4]
Explain why we request blocks that are referenced by an incoming block but that we don't have in our chain. Why don't we just ignore them?

Solution:
In a blockchain network, it's crucial to request blocks referenced by an incoming block but not present in our chain for several reasons:

1. Chain Consistency: Requesting missing blocks helps ensure that our local copy of the blockchain remains consistent with the rest of the network. If we ignore missing blocks, our chain may become incomplete or outdated, leading to inconsistencies with other nodes.
2. Blockchain Validation: Each block in a blockchain references the hash of the previous block. When receiving a new block, if we don't have the referenced previous block, we can't validate the new block's integrity. Requesting missing blocks allows us to validate each block in the chain properly.
3. Fork Resolution: Ignoring missing blocks could lead to forks in the blockchain. If different nodes have different portions of the blockchain due to missing blocks, it may result in a split in the network. By requesting missing blocks, we help prevent forks by ensuring that all nodes have access to the same chain of blocks.
4. Data Integrity: Blockchain protocols often rely on the assumption that all nodes have access to the same set of blocks. Ignoring missing blocks could lead to inconsistencies in the data stored on different nodes, undermining the integrity and trustworthiness of the blockchain.
5. Network Health: Requesting missing blocks also helps identify potential issues in the network, such as block propagation delays or network partitions. If we consistently fail to receive certain blocks, it may indicate problems with specific nodes or network segments that need to be addressed.

Overall, requesting blocks referenced by an incoming block but not present in our chain is essential for maintaining chain consistency, validating block integrity, resolving forks, preserving data integrity, and ensuring the health and reliability of the blockchain network.

##### [C5]
Our blockchain works with a notion of time slots. Why should we expect consensus about when these slots start and end?

**HINT**: Thomas Schelling.

Solution:
We should expect consensus about when time slots start and end in a blockchain because of the concept of coordination games, as described by Thomas Schelling. He studied how individuals achieve coordination and cooperation in situations where there is no central authority. One of his key insights is that people often coordinate their actions based on focal points or salient features of the situation, rather than explicit communication or coordination mechanisms.

In the context of blockchain and time slots:

1. Focal Points: The notion of time slots provides a focal point for coordination. Even without explicit communication or coordination mechanisms, participants in the blockchain network can use the concept of time slots as a natural reference point for organizing their actions. This helps to achieve consensus on when transactions should occur, when blocks should be produced, and when other blockchain-related activities should take place.
2. Common Understanding: Over time, participants in the blockchain network develop a common understanding of how time slots are defined and when they start and end. This shared understanding becomes a focal point for coordinating their activities and ensuring that the blockchain operates smoothly.
3. Consensus Emerges: Through repeated interactions and observations of others' behavior, consensus gradually emerges about the timing of time slots. Participants adjust their behavior to align with this consensus, reinforcing the focal point and further solidifying coordination.
4. Security and Reliability: Consensus about time slots enhances the security and reliability of the blockchain network. It ensures that transactions are processed in a timely manner, blocks are produced at regular intervals, and the overall operation of the blockchain is predictable and stable.

Therefore, consensus about when time slots start and end in a blockchain emerges through the use of focal points and coordination mechanisms, as described by Thomas Schelling. This consensus helps to organize and synchronize activities within the blockchain network, contributing to its security, reliability, and efficiency.
